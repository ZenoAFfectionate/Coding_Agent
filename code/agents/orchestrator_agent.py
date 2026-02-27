"""OrchestratorAgent — master agent that delegates to specialized workers.

Inherits from ``Agent`` to reuse context budget management,
token counting, trajectory tracking, and debug printing.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

from code.core.agent import Agent
from code.core.config import Config
from code.core.llm import HelloAgentsLLM
from code.agents.function_call_agent import FunctionCallAgent
from code.tools.builtin.finish_tool import FinishTool
from code.tools.builtin.escalate_tool import EscalateTool
from code.tools.registry import ToolRegistry

from code.swarm.blackboard import Blackboard
from code.swarm.paths import load_swarm_prompt
from code.swarm.worker_factory import WORKER_SPECS, _build_tools

logger = logging.getLogger("multi_agent")


# ---------------------------------------------------------------------------
# Orchestrator tool schemas
# ---------------------------------------------------------------------------

def _build_orchestrator_tools() -> list[dict]:
    """Build the OpenAI-compatible function-calling tool schemas."""
    return [
        {
            "type": "function",
            "function": {
                "name": "dispatch_worker",
                "description": (
                    "Dispatch a task to a specialist worker agent. "
                    "The task string must be completely self-contained — "
                    "include all context the worker needs."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "worker": {
                            "type": "string",
                            "enum": list(WORKER_SPECS.keys()),
                            "description": "The specialist worker type.",
                        },
                        "task": {
                            "type": "string",
                            "description": (
                                "Self-contained task instruction including "
                                "file paths, prior findings, and all context."
                            ),
                        },
                    },
                    "required": ["worker", "task"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finish",
                "description": (
                    "Provide the final synthesized answer to the user. "
                    "Call this when you have enough information to fully "
                    "address the request."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The complete answer for the user.",
                        },
                    },
                    "required": ["answer"],
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# OrchestratorAgent
# ---------------------------------------------------------------------------

class OrchestratorAgent(Agent):
    """Master agent that delegates to specialized workers.

    Inherits from ``Agent`` to reuse context budget management,
    token counting, trajectory tracking, and debug printing.

    Supports two communication modes with the LLM:
    1. **Function calling** (default) — uses OpenAI-compatible tool schemas
       ``dispatch_worker`` and ``finish`` for structured, reliable output.
    2. **Text fallback** — if the provider does not support function calling,
       automatically falls back to JSON-in-text parsing.

    Also includes:
    - Worker result truncation to cap per-result size.
    - Error detection for worker failures, surfaced explicitly to the LLM
      so it can decide to retry or adjust.
    """

    # Patterns that indicate a worker result is an error.
    _WORKER_ERROR_PATTERNS = [
        "failed with error:",
        "unable to complete the task within the allowed",
        "traceback (most recent call last)",
        "error executing python code:",
        "error executing shell command:",
        "error: execution timed out",
        "segmentation fault",
        "killed",
    ]

    # Known benign stderr patterns to exclude from error detection.
    _BENIGN_PATTERNS = [
        "error processing line",       # broken .pth files
        "remainder of file ignored",   # continuation of .pth errors
        "deprecationwarning",          # third-party deprecation warnings
        "[notice] a new release of pip",  # pip upgrade notices
    ]

    def __init__(
        self,
        workspace: str,
        config: Config,
        max_worker_steps: int = 10,
        max_orchestrator_rounds: int = 16,
        max_result_chars: int = 4000,
        context_max_tokens: int = 0,
        enable_summarization: bool = True,
        enable_reflection: bool = True,
        debug: bool = True,
    ):
        # Build LLM first so we can pass it to Agent.__init__
        llm = HelloAgentsLLM(
            model=config.default_model,
            provider=config.default_provider,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        # Ensure config.debug matches the explicit debug flag
        config.debug = debug

        super().__init__(
            name="Orchestrator",
            llm=llm,
            config=config,
            context_max_tokens=context_max_tokens,
        )

        # Override Agent's list[Message] history with dict-based history
        # (the orchestrator stores {"role": ..., "content": ...} dicts)
        self._history: list[dict] = []

        self.workspace = str(Path(workspace).resolve())
        self.max_worker_steps = max_worker_steps
        self.max_orchestrator_rounds = max_orchestrator_rounds
        self.max_result_chars = max_result_chars
        self.enable_summarization = enable_summarization
        self.enable_reflection = enable_reflection

        # Cache for tool registries — avoids rebuilding on every worker dispatch
        self._tool_registry_cache: dict[str, ToolRegistry] = {}

        # Function calling: enabled by default, auto-disabled on first failure
        self._use_function_calling = True
        self._orchestrator_tools = _build_orchestrator_tools()

        # Build both prompt variants (FC and text-mode)
        worker_descs = "\n".join(
            f"- **{wtype}**: {spec['description']}"
            for wtype, spec in WORKER_SPECS.items()
        )
        worker_types = ", ".join(WORKER_SPECS.keys())

        fc_template = load_swarm_prompt("orchestrator_fc")
        self._fc_system_prompt = fc_template.format(
            worker_descriptions=worker_descs,
        )

        text_template = load_swarm_prompt("orchestrator")
        self._text_system_prompt = text_template.format(
            worker_descriptions=worker_descs,
            worker_types=worker_types,
        )

        # Summarization prompt
        self._summarize_prompt = load_swarm_prompt("summarize_result")

        # Reflection prompt
        self._reflection_prompt = load_swarm_prompt("reflection_orchestrator")

    # ------------------------------------------------------------------ #
    #  system_prompt property (overrides Agent's plain attribute)
    # ------------------------------------------------------------------ #

    @property
    def system_prompt(self) -> str:
        """Return the active system prompt (depends on current mode)."""
        return self._fc_system_prompt if self._use_function_calling else self._text_system_prompt

    @system_prompt.setter
    def system_prompt(self, _value):
        # Accept writes from Agent.__init__ without error; the orchestrator
        # manages its own prompt variants (_fc_system_prompt / _text_system_prompt).
        pass

    # ------------------------------------------------------------------ #
    #  Worker result truncation & error detection
    # ------------------------------------------------------------------ #

    def _truncate_result(self, result: str) -> str:
        """Truncate a worker result to *max_result_chars* (middle-cut)."""
        if len(result) <= self.max_result_chars:
            return result
        half = self.max_result_chars // 2
        omitted = len(result) - self.max_result_chars
        return (
            result[:half]
            + f"\n\n... [{omitted} chars truncated] ...\n\n"
            + result[-half:]
        )

    def _detect_worker_status(self, result: str) -> str:
        """Classify worker result as "success", "incomplete", or "error".

        Filters out known benign stderr patterns before checking for error
        indicators. Distinguishes between hard errors and incomplete results
        (e.g., max steps exhausted, circuit breaker, escalation).
        """
        lower = result.lower()

        # Check for escalation or circuit breaker (incomplete, not error)
        if result.startswith("[ESCALATED]") or "[circuit breaker tripped" in lower:
            return "incomplete"
        if "max steps exhausted" in lower:
            return "incomplete"

        # Strip lines matching benign patterns before checking
        filtered_lines = []
        for line in lower.splitlines():
            if not any(bp in line for bp in self._BENIGN_PATTERNS):
                filtered_lines.append(line)
        filtered = "\n".join(filtered_lines)

        if any(p in filtered for p in self._WORKER_ERROR_PATTERNS):
            return "error"

        return "success"

    # ------------------------------------------------------------------ #
    #  Worker result summarization
    # ------------------------------------------------------------------ #

    def _summarize_result(self, worker_type: str, task: str, result: str) -> str:
        """Summarize a long worker result via LLM; short results pass through."""
        if not self.enable_summarization or len(result) <= self.max_result_chars:
            return result
        if not self._summarize_prompt:
            return self._truncate_result(result)
        try:
            summary = self.llm.invoke([
                {"role": "system", "content": self._summarize_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Worker: {worker_type}\n"
                        f"Task: {task}\n\n"
                        f"Full result:\n{result}"
                    ),
                },
            ])
            if summary:
                return summary
        except Exception as e:
            self._print(f"  [Summarization failed: {e}] — falling back to truncation.")
        return self._truncate_result(result)

    # ------------------------------------------------------------------ #
    #  Reflection / self-verification
    # ------------------------------------------------------------------ #

    def _reflect_on_answer(self, user_request: str, blackboard: Blackboard, answer: str) -> str:
        """Run a quality-check LLM call on the proposed final answer.

        Only replaces the answer if the reflection explicitly provides a
        revised version. If the reflection output is ambiguous or fails,
        the original answer is preserved.
        """
        if not self._reflection_prompt:
            return answer
        prompt = self._reflection_prompt.format(
            user_request=user_request,
            blackboard=blackboard.serialize(),
            proposed_answer=answer,
        )
        try:
            reflection = self.llm.invoke([
                {"role": "system", "content": "You are a quality reviewer."},
                {"role": "user", "content": prompt},
            ])
            if not reflection or not reflection.strip():
                self._print("\n[Reflection] Empty response — keeping original answer.", level="info")
                return answer
            stripped = reflection.strip()
            if stripped == "APPROVED":
                self._print("\n[Reflection] Answer approved.", level="info")
                return answer
            # Only accept revision if it's substantively different and longer
            # than a trivial response (avoids replacing with "NEEDS REVISION" etc.)
            if len(stripped) > 50 and len(stripped) > len(answer) * 0.3:
                self._print("\n[Reflection] Answer revised by reflection step.", level="info")
                logger.info("[Reflection] Revised. Original len=%d, Revised len=%d",
                            len(answer), len(stripped))
                return stripped
            self._print("\n[Reflection] Reflection too short to be a valid revision — keeping original.", level="info")
        except Exception as e:
            self._print(f"\n[Reflection failed: {e}] — returning original answer.", level="info")
        return answer

    # ------------------------------------------------------------------ #
    #  Run a single worker
    # ------------------------------------------------------------------ #

    def _run_worker(self, worker_type: str, task: str) -> tuple[str, str]:
        """Spawn a worker, run the task, return ``(result, status)``.

        *status* is one of ``"success"``, ``"incomplete"``, or ``"error"``.
        Caches tool registries per worker type to avoid repeated construction.
        """
        if worker_type not in WORKER_SPECS:
            logger.error("Unknown worker type: %s", worker_type)
            return f"Error: unknown worker type '{worker_type}'", True

        self._print(f"\n{'='*50}")
        self._print(f"  [Dispatching] Worker: {worker_type}")
        self._print(f"  [Task] {task[:200]}{'...' if len(task) > 200 else ''}")
        self._print(f"{'='*50}")
        logger.info("[Dispatch] worker=%s task=%.500s", worker_type, task)

        # Use cached tool registry if available
        spec = WORKER_SPECS[worker_type]
        if worker_type not in self._tool_registry_cache:
            registry = _build_tools(self.workspace, spec["tools"])
            registry.register_tool(FinishTool())
            registry.register_tool(EscalateTool())
            self._tool_registry_cache[worker_type] = registry

        base_system = load_swarm_prompt("system") or "You are an expert Python coding assistant."
        task_prompt = load_swarm_prompt(spec["prompt"])
        system_prompt = base_system
        if task_prompt:
            system_prompt += f"\n\n<task-instructions>\n{task_prompt}\n</task-instructions>"

        worker = FunctionCallAgent(
            name=f"Worker_{worker_type}",
            llm=self.llm,
            system_prompt=system_prompt,
            tool_registry=self._tool_registry_cache[worker_type],
            max_steps=self.max_worker_steps,
            config=self.config,
        )

        t0 = time.time()
        try:
            result = worker.run(task)
        except Exception as e:
            result = f"Worker '{worker_type}' failed with error: {e}"
            logger.error("[Worker %s] Exception: %s", worker_type, e, exc_info=True)

        elapsed = time.time() - t0
        self._print(f"\n  [Worker {worker_type} done] Result length: {len(result)} chars")
        status = self._detect_worker_status(result)
        if status == "error":
            self._print(f"  [Worker {worker_type}] Warning: Error detected in result.", level="info")
        elif status == "incomplete":
            self._print(f"  [Worker {worker_type}] Worker returned incomplete result.", level="info")
        logger.info("[Worker %s] done in %.1fs, len=%d, status=%s, result=%.1000s",
                     worker_type, elapsed, len(result), status, result)
        return result, status

    # ------------------------------------------------------------------ #
    #  Orchestrator LLM invocation (unified dispatcher)
    # ------------------------------------------------------------------ #

    def _invoke_orchestrator(
        self,
        messages: list[dict],
        force_finish: bool = False,
    ) -> tuple[str, object, list[dict]]:
        """Call the orchestrator LLM and interpret its decision.

        Returns:
            ``(action, data, msgs_to_append)`` where:
            - *action* is ``"dispatch"`` or ``"finish"``.
            - *data* is ``{"worker": str, "task": str, "tool_call_id": str}``
              for dispatch, or a plain string for finish.
            - *msgs_to_append* is a list of message dicts to extend the
              conversation with (assistant response + optional forced prompt).
        """
        if self._use_function_calling:
            try:
                return self._invoke_fc(messages, force_finish)
            except Exception as exc:
                self._print(
                    f"\n  [Orchestrator] Function calling failed ({exc}), "
                    "switching to text mode for the rest of this session.",
                    level="info",
                )
                self._use_function_calling = False
        return self._invoke_text(messages, force_finish)

    # ---- Function-calling path ----------------------------------------

    def _invoke_fc(
        self,
        messages: list[dict],
        force_finish: bool = False,
    ) -> tuple[str, object, list[dict]]:
        """Function-calling path: returns ``(action, data, msgs)``."""
        tool_choice: str | dict = "auto"
        if force_finish:
            tool_choice = {
                "type": "function",
                "function": {"name": "finish"},
            }

        response = self.llm.invoke_with_tools(
            messages,
            tools=self._orchestrator_tools,
            tool_choice=tool_choice,
        )
        msg = response.choices[0].message

        # Build the assistant dict for conversation history
        assistant_msg: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]

        if msg.tool_calls:
            tc = msg.tool_calls[0]
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                # Malformed arguments — treat as finish with raw text
                return "finish", tc.function.arguments, [assistant_msg]

            if tc.function.name == "finish":
                return "finish", args.get("answer", ""), [assistant_msg]

            if tc.function.name == "dispatch_worker":
                data = {
                    "worker": args.get("worker", ""),
                    "task": args.get("task", ""),
                    "tool_call_id": tc.id,
                }
                return "dispatch", data, [assistant_msg]

        # No tool call — treat raw text as the final answer
        return "finish", msg.content or "", [assistant_msg]

    # ---- Text-mode fallback path --------------------------------------

    def _invoke_text(
        self,
        messages: list[dict],
        force_finish: bool = False,
    ) -> tuple[str, object, list[dict]]:
        """Text-based fallback: returns ``(action, data, msgs)``."""
        call_messages = list(messages)
        extra_msgs: list[dict] = []
        if force_finish:
            force_msg = {
                "role": "user",
                "content": (
                    "You have reached the maximum number of rounds. "
                    "Provide your final answer now using: "
                    '{"done": true, "answer": "..."}'
                ),
            }
            call_messages.append(force_msg)
            extra_msgs.append(force_msg)

        response_text = self.llm.invoke(call_messages) or ""
        assistant_msg = {"role": "assistant", "content": response_text}
        self._print(f"\n[Orchestrator] Response:\n{response_text}")

        dispatch = self._parse_single_dispatch(response_text)
        if dispatch is None:
            answer = self._extract_done_answer(response_text)
            return "finish", answer, extra_msgs + [assistant_msg]

        data = {
            "worker": dispatch.get("worker", ""),
            "task": dispatch.get("task", ""),
            "tool_call_id": "",
        }
        return "dispatch", data, extra_msgs + [assistant_msg]

    # ---- Text-mode JSON parsers (kept as fallback) --------------------

    @staticmethod
    def _parse_single_dispatch(response: str) -> Optional[dict]:
        """Extract a single ``{"worker":..,"task":..}`` dispatch from text.

        Returns the dict or None (done / no JSON found).
        """
        def _check(obj: object) -> Optional[dict]:
            if not isinstance(obj, dict):
                return None
            if obj.get("done"):
                return None
            if "worker" in obj and "task" in obj:
                return obj
            return None

        # Tier 1: fenced ```json blocks
        for block in re.findall(r'```json\s*\n?(.*?)```', response, re.DOTALL):
            try:
                parsed = json.loads(block.strip())
                result = _check(parsed)
                if result is not None:
                    return result
                if isinstance(parsed, dict):
                    return None
            except json.JSONDecodeError:
                continue

        # Tier 2: entire response as JSON
        try:
            parsed = json.loads(response.strip())
            result = _check(parsed)
            if result is not None:
                return result
            if isinstance(parsed, dict):
                return None
        except json.JSONDecodeError:
            pass

        # Tier 3: find the first balanced JSON object in the response
        first_brace = response.find("{")
        if first_brace != -1:
            candidate = OrchestratorAgent._extract_balanced_braces(response, first_brace)
            if candidate:
                try:
                    parsed = json.loads(candidate)
                    result = _check(parsed)
                    if result is not None:
                        return result
                    if isinstance(parsed, dict):
                        return None
                except json.JSONDecodeError:
                    pass

        return None

    @staticmethod
    def _extract_done_answer(response: str) -> str:
        """Pull the ``answer`` field from ``{"done":true,"answer":"..."}``."""
        def _try(obj: object) -> Optional[str]:
            if isinstance(obj, dict) and obj.get("done") and "answer" in obj:
                return obj["answer"]
            return None

        for block in re.findall(r'```json\s*\n?(.*?)```', response, re.DOTALL):
            try:
                r = _try(json.loads(block.strip()))
                if r is not None:
                    return r
            except json.JSONDecodeError:
                continue

        try:
            r = _try(json.loads(response.strip()))
            if r is not None:
                return r
        except json.JSONDecodeError:
            pass

        first_brace = response.find("{")
        if first_brace != -1:
            candidate = OrchestratorAgent._extract_balanced_braces(response, first_brace)
            if candidate:
                try:
                    r = _try(json.loads(candidate))
                    if r is not None:
                        return r
                except json.JSONDecodeError:
                    pass

        return response

    @staticmethod
    def _extract_balanced_braces(text: str, start: int) -> Optional[str]:
        """Extract a balanced ``{...}`` substring starting at *start*.

        Correctly handles nested braces and JSON string literals,
        unlike the previous ``r'\\{[^{}]*\\}'`` regex which broke on
        nested objects.
        """
        if start >= len(text) or text[start] != "{":
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\" and in_string:
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #

    def run(self, user_input: str) -> str:
        """Process a user request through the iterative multi-agent pipeline."""
        self._print(f"\n[Orchestrator] Received: {user_input}", level="info")
        logger.info("=" * 80)
        logger.info("[Orchestrator] NEW REQUEST: %.500s", user_input)

        blackboard = Blackboard(user_input)

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self._history)
        messages.append({"role": "user", "content": user_input})

        final_answer: Optional[str] = None

        for round_num in range(1, self.max_orchestrator_rounds + 1):
            self._print(f"\n[Orchestrator] Round {round_num}/{self.max_orchestrator_rounds}...")
            logger.info("[Orchestrator] Round %d/%d", round_num, self.max_orchestrator_rounds)

            # Apply context budget before each LLM call
            trimmed = self._manage_context_budget(messages)

            action, data, to_append = self._invoke_orchestrator(trimmed)
            messages.extend(to_append)
            logger.info("[Orchestrator] Decision: action=%s data=%.500s", action, str(data))

            # --- Finish ---
            if action == "finish":
                final_answer = data
                self._print("\n[Orchestrator] Done — no more workers to dispatch.", level="info")
                logger.info("[Orchestrator] FINISH — answer_len=%d", len(str(final_answer)))
                break

            # --- Dispatch ---
            worker_type = data["worker"].strip()
            task = data["task"].strip()
            tool_call_id = data.get("tool_call_id", "")

            if not worker_type or not task:
                self._print(
                    "\n[Orchestrator] Invalid dispatch (missing worker/task), treating as done.",
                    level="info",
                )
                logger.warning("[Orchestrator] Invalid dispatch: %s", data)
                final_answer = str(data)
                break

            self._print(f"\n--- Round {round_num}: dispatching [{worker_type}] ---", level="info")
            logger.info("[Orchestrator] Dispatching [%s] with task: %.500s", worker_type, task)
            result, status = self._run_worker(worker_type, task)

            # Summarize (or truncate) long results
            result = self._summarize_result(worker_type, task, result)

            # Scan workspace for created files after each worker run
            blackboard.scan_workspace_files(self.workspace)

            # Update the blackboard (store more context — 1500 chars)
            if status == "error":
                blackboard.add_error(worker_type, round_num, result[:1500])
            else:
                blackboard.add_finding(worker_type, round_num, result[:1500])
            logger.info("[Blackboard] State after round %d:\n%s", round_num, blackboard.serialize())

            # Build the feedback message with blackboard state
            bb_section = f"\n\n<blackboard>\n{blackboard.serialize()}\n</blackboard>"
            if status == "error":
                feedback = (
                    f"[WORKER ERROR] Worker `{worker_type}` encountered an error:\n\n"
                    f"{result}\n\n"
                    "You may retry with adjusted parameters, try a different worker, "
                    "or provide the final answer with the information gathered so far."
                    f"{bb_section}"
                )
            elif status == "incomplete":
                feedback = (
                    f"[WORKER INCOMPLETE] Worker `{worker_type}` could not finish the task:\n\n"
                    f"{result}\n\n"
                    "Options: (1) re-dispatch with more context or a simpler sub-task, "
                    "(2) try a different worker type, or (3) call finish() with the best answer so far."
                    f"{bb_section}"
                )
            else:
                feedback = (
                    f"Worker `{worker_type}` completed successfully.\n\n"
                    f"**Task:** {task}\n\n"
                    f"**Result:**\n{result}"
                    f"{bb_section}"
                )

            # Append feedback in the correct role for the active mode
            if tool_call_id:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": feedback,
                })
            else:
                messages.append({
                    "role": "user",
                    "content": feedback + "\n\nDecide your next action.",
                })
        else:
            # Round limit exhausted — force a final synthesis
            self._print(
                f"\n[Orchestrator] Round limit ({self.max_orchestrator_rounds}) reached, "
                "forcing synthesis...",
                level="info",
            )
            trimmed = self._manage_context_budget(messages)
            action, data, to_append = self._invoke_orchestrator(trimmed, force_finish=True)
            messages.extend(to_append)
            final_answer = data if action == "finish" else str(data)

        if final_answer is None:
            final_answer = "Unable to produce a final answer."

        # Fallback: if the orchestrator returned an empty answer, synthesize
        # from the blackboard findings so the user gets *something* useful.
        if not final_answer.strip():
            self._print("\n[Orchestrator] Empty answer detected — synthesizing from blackboard.", level="info")
            logger.warning("[Orchestrator] Empty final answer, synthesizing from blackboard.")
            bb_state = blackboard.serialize()
            if blackboard.findings or blackboard.errors:
                final_answer = (
                    "The analysis produced the following results:\n\n"
                    + bb_state
                )
                if blackboard.files_created:
                    final_answer += (
                        "\n\nGenerated files: "
                        + ", ".join(blackboard.files_created)
                    )
            else:
                final_answer = "No actionable results were produced. Please try rephrasing the request."

        # Reflection: quality-check the final answer before returning
        if self.enable_reflection and final_answer:
            final_answer = self._reflect_on_answer(user_input, blackboard, final_answer)

        # Save to conversation history (mode-agnostic: just user + answer)
        self._history.append({"role": "user", "content": user_input})
        self._history.append({"role": "assistant", "content": final_answer})

        return final_answer
