"""
BFCL Evaluator Module

Responsible for evaluating the agent's performance on the BFCL benchmark.
"""

from typing import Dict, Any, List, Optional, Union
import json
import ast
import logging
import sys
import time
from pathlib import Path
from code.evaluation.benchmarks.bfcl.dataset import BFCLDataset
from code.evaluation.benchmarks.bfcl.metrics import BFCLMetrics

logger = logging.getLogger(__name__)

# Evaluation prompts directory (benchmark-specific prompts)
EVAL_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

_MULTI_TURN_TOOL_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the provided tools to complete the user's tasks."
)


def _load_prompt(path: Path, fallback: str | None = None) -> str | None:
    """Load a prompt file, returning *fallback* if the file is missing or empty."""
    try:
        text = path.read_text(encoding="utf-8").strip()
        return text or fallback
    except FileNotFoundError:
        logger.warning("Prompt file not found: %s", path)
        return fallback


class BFCLEvaluator:
    """BFCL Evaluator

    Evaluates the agent's tool-calling capabilities, including:
    - Simple function calls
    - Multiple function calls
    - Parallel function calls
    - Irrelevance detection

    Supports two evaluation modes:
    - AST evaluation: Abstract Syntax Tree matching
    - Execution evaluation: Actual function execution result comparison

    Attributes:
        dataset: BFCL dataset
        metrics: Evaluation metrics calculator
        evaluation_mode: Evaluation mode ('ast' or 'execution')
    """

    def __init__(
        self,
        dataset: Optional[BFCLDataset] = None,
        category: Optional[str] = None,
        evaluation_mode: str = "ast",
        local_data_dir: Optional[str] = None,
        llm=None,
    ):
        """Initialize the BFCL evaluator.

        Args:
            dataset: BFCL dataset. If None, one will be created automatically.
            category: Evaluation category.
            evaluation_mode: Evaluation mode ('ast' or 'execution').
            local_data_dir: Local data directory.
            llm: Optional HelloAgentsLLM instance for direct LLM mode
                 (bypasses the ReAct agent loop).
        """
        self.dataset = dataset or BFCLDataset(
            category=category,
            local_data_dir=local_data_dir
        )
        self.metrics = BFCLMetrics()
        self.evaluation_mode = evaluation_mode
        self.category = category
        self.llm = llm

        # Load benchmark-specific system prompt
        self.bfcl_system_prompt = _load_prompt(EVAL_PROMPTS_DIR / "bfcl_system.prompt")

    def evaluate(self, agent: Any = None, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate using either a direct LLM or a ReAct agent.

        When ``self.llm`` is set the evaluator calls the LLM directly (single-shot),
        which is much faster and avoids the ReAct system prompt conflicting with the
        BFCL prompt.  Otherwise it falls back to ``agent.run()``.

        Args:
            agent: The agent to evaluate (used only when ``self.llm`` is None).
            max_samples: Maximum number of samples to evaluate. None means evaluate all.

        Returns:
            Evaluation result dictionary containing various metrics.
        """
        mode_label = "direct LLM" if self.llm else "agent"
        agent_name = getattr(self.llm, 'model', None) or getattr(agent, 'name', 'Unknown')

        print(f"\n🔧 Starting BFCL evaluation...")
        print(f"   Mode: {mode_label}")
        print(f"   Model / Agent: {agent_name}")
        print(f"   Evaluation mode: {self.evaluation_mode}")
        print(f"   Category: {self.category or 'All'}")

        # Load dataset
        dataset = self.dataset.load()
        if not dataset:
            print("   ⚠️ Dataset is empty, skipping evaluation")
            return self._create_empty_results(agent)

        # Limit sample count
        if max_samples:
            dataset = dataset[:max_samples]

        # Detect multi-turn mode
        is_multi_turn = False
        if dataset:
            q = dataset[0].get("question", "")
            if isinstance(q, list) and q and isinstance(q[0], list):
                is_multi_turn = True
                num_turns = len(q)
                print(f"   Multi-turn: yes ({num_turns} turns per sample)")

        print(f"   Sample count: {len(dataset)}")

        # Run evaluation
        results = []
        categories = {}
        total = len(dataset)
        bar_width = 30
        is_tty = sys.stdout.isatty()

        for i, sample in enumerate(dataset):
            done = i + 1
            correct_so_far = sum(1 for r in results if r.get("success"))

            if is_tty:
                # Interactive terminal: overwrite progress bar in place
                filled = int(bar_width * done / total)
                bar = "█" * filled + "░" * (bar_width - filled)
                pct = done / total * 100
                print(f"\r   [{bar}] {done}/{total} ({pct:.0f}%) | ✓ {correct_so_far}", end="", flush=True)
            elif done % 10 == 0 or done == total:
                # Non-TTY (piped/redirected): print periodic one-line updates
                print(f"   Progress: {done}/{total} | correct: {correct_so_far}")

            try:
                sample_result = self.evaluate_sample(agent, sample)
                results.append(sample_result)

                # Aggregate by category (use the evaluator's category, not the sample's category)
                category = self.category if self.category else sample.get("category", "unknown")
                if category not in categories:
                    categories[category] = {"total": 0, "correct": 0, "results": []}

                categories[category]["total"] += 1
                if sample_result["success"]:
                    categories[category]["correct"] += 1
                categories[category]["results"].append(sample_result)

            except Exception as e:
                print(f"   ⚠️ Sample {i} evaluation failed: {e}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "predicted": None,
                    "expected": sample.get("ground_truth"),
                    "score": 0.0
                })

        print()  # Newline after progress bar

        # Calculate overall metrics
        total_samples = len(results)
        correct_samples = sum(1 for r in results if r["success"])
        overall_accuracy = correct_samples / total_samples if total_samples > 0 else 0.0

        # Calculate per-category metrics
        category_metrics = {}
        for cat, cat_data in categories.items():
            accuracy = cat_data["correct"] / cat_data["total"] if cat_data["total"] > 0 else 0.0
            category_metrics[cat] = {
                "total": cat_data["total"],
                "correct": cat_data["correct"],
                "accuracy": accuracy
            }

        final_results = {
            "benchmark": "BFCL",
            "agent_name": agent_name,
            "evaluation_mode": self.evaluation_mode,
            "category": self.category,
            "total_samples": total_samples,
            "correct_samples": correct_samples,
            "overall_accuracy": overall_accuracy,
            "category_metrics": category_metrics,
            "detailed_results": results
        }

        print(f"✅ BFCL evaluation completed")
        print(f"   Overall accuracy: {overall_accuracy:.2%}")
        for cat, metrics in category_metrics.items():
            print(f"   {cat}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")

        return final_results

    def evaluate_sample(self, agent: Any, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single sample.

        Uses direct LLM invocation when ``self.llm`` is set, otherwise falls
        back to ``agent.run()``.

        Args:
            agent: The agent to evaluate (ignored when ``self.llm`` is set).
            sample: Sample data.

        Returns:
            Evaluation result for a single sample.
        """
        # Detect multi-turn samples: question is List[List[Dict]]
        question = sample.get("question", "")
        if isinstance(question, list) and question and isinstance(question[0], list):
            return self._evaluate_multi_turn_sample(sample)

        try:
            # Prepare input (single-turn)
            functions = sample.get("function", [])
            ground_truth = sample.get("ground_truth", [])

            # Build function-calling prompt
            prompt = self._build_function_calling_prompt(question, functions)

            # Call the LLM directly or via agent
            start_time = time.time()
            if self.llm:
                messages = []
                if self.bfcl_system_prompt:
                    messages.append({"role": "system", "content": self.bfcl_system_prompt})
                messages.append({"role": "user", "content": prompt})
                response = self.llm.invoke(messages)
            else:
                response = agent.run(prompt)
            execution_time = time.time() - start_time

            # Parse function calls from the response
            predicted_calls = self._extract_function_calls(response)

            # Evaluate results
            if self.evaluation_mode == "ast":
                success, score = self._evaluate_ast_matching(predicted_calls, ground_truth)
            else:
                success, score = self._evaluate_execution(predicted_calls, ground_truth, functions)

            return {
                "success": success,
                "score": score,
                "predicted": predicted_calls,
                "expected": ground_truth,
                "response": response,
                "question": question,
                "execution_time": execution_time,
                "sample_id": sample.get("id", ""),
                "category": self.category if self.category else sample.get("category", "unknown")
            }

        except Exception as e:
            return {
                "success": False,
                "score": 0.0,
                "predicted": None,
                "expected": sample.get("ground_truth", []),
                "question": sample.get("question", ""),
                "error": str(e),
                "sample_id": sample.get("id", ""),
                "category": self.category if self.category else sample.get("category", "unknown")
            }

    # ------------------------------------------------------------------
    # Multi-turn evaluation
    # ------------------------------------------------------------------

    # Mapping from BFCL class names to function-doc filenames
    _CLASS_TO_DOC_FILE = {
        "GorillaFileSystem": "gorilla_file_system.json",
        "MathAPI": "math_api.json",
        "MessageAPI": "message_api.json",
        "TicketAPI": "ticket_api.json",
        "TradingBot": "trading_bot.json",
        "TravelAPI": "travel_booking.json",
        "TwitterAPI": "posting_api.json",
        "VehicleControlAPI": "vehicle_control.json",
    }

    @staticmethod
    def _convert_bfcl_schema(schema: dict) -> dict:
        """Recursively convert a BFCL parameter schema to OpenAI JSON Schema."""
        if not isinstance(schema, dict):
            return schema

        out = dict(schema)

        # Type mappings
        t = out.get("type")
        if t == "dict":
            out["type"] = "object"
        elif t == "float":
            out["type"] = "number"

        # Array without items
        if out.get("type") == "array" and "items" not in out:
            out["items"] = {"type": "string"}

        # Recurse into properties
        if "properties" in out:
            out["properties"] = {
                k: BFCLEvaluator._convert_bfcl_schema(v)
                for k, v in out["properties"].items()
            }

        # Recurse into items
        if "items" in out and isinstance(out["items"], dict):
            out["items"] = BFCLEvaluator._convert_bfcl_schema(out["items"])

        return out

    def _bfcl_to_openai_tools(self, func_docs: List[Dict]) -> List[Dict]:
        """Convert BFCL func_doc list to OpenAI tool-calling schema."""
        tools = []
        for doc in func_docs:
            params = doc.get("parameters", {})
            converted = self._convert_bfcl_schema(params)
            # Strip the "response" field if present — not part of OpenAI schema
            func_schema = {
                "name": doc.get("name", ""),
                "description": doc.get("description", ""),
                "parameters": converted,
            }
            tools.append({"type": "function", "function": func_schema})
        return tools

    @staticmethod
    def _build_func_param_lookup(func_docs: List[Dict]) -> Dict[str, List[str]]:
        """Build ``{func_name: [param1, param2, ...]}`` from func_docs.

        Python 3.7+ preserves dict insertion order, so the parameter order
        matches BFCL's declaration order.
        """
        lookup: dict[str, list[str]] = {}
        for doc in func_docs:
            name = doc.get("name", "")
            props = doc.get("parameters", {}).get("properties", {})
            lookup[name] = list(props.keys())
        return lookup

    @staticmethod
    def _normalize_call_to_keyword_args(
        call_str: str,
        func_params_lookup: Dict[str, List[str]],
    ) -> str:
        """Convert positional args to keyword args in a function-call string.

        E.g. ``sort('file.pdf')`` with lookup ``{"sort": ["file_name"]}``
        becomes ``sort(file_name='file.pdf')``.
        """
        try:
            tree = ast.parse(call_str, mode="eval")
        except SyntaxError:
            return call_str

        node = tree.body
        if not isinstance(node, ast.Call):
            return call_str

        # Determine function name
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        else:
            return call_str

        param_names = func_params_lookup.get(func_name)
        if param_names is None:
            return call_str

        # Convert positional args to keyword args
        new_keywords = list(node.keywords)  # existing keywords
        for idx, arg in enumerate(node.args):
            if idx < len(param_names):
                new_keywords.append(ast.keyword(arg=param_names[idx], value=arg))

        node.args = []
        node.keywords = new_keywords

        return ast.unparse(tree)

    def _evaluate_multi_turn_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a multi-turn BFCL sample using native tool calling.

        Multi-turn samples have:
          - question: List[List[Dict]]  (turns of user messages)
          - ground_truth: List[List[str]]  (per-turn function-call strings)
          - involved_classes / path / excluded_function instead of 'function'
        """
        turns = sample["question"]                    # List[List[message_dict]]
        ground_truth = sample.get("ground_truth", [])  # List[List[str]]

        # Load function definitions from multi_turn_func_doc/
        functions = self._load_multi_turn_functions(sample)

        # Convert to OpenAI tools and build param lookup for AST normalisation
        tools = self._bfcl_to_openai_tools(functions)
        func_params_lookup = self._build_func_param_lookup(functions)

        conversation: list[dict] = [
            {"role": "system", "content": _MULTI_TURN_TOOL_SYSTEM_PROMPT},
        ]

        turn_results = []
        all_correct = True
        start_time = time.time()

        for turn_idx, turn_messages in enumerate(turns):
            turn_gt = ground_truth[turn_idx] if turn_idx < len(ground_truth) else []

            # If the turn has no user messages (e.g. miss_func re-introduction)
            if not turn_messages:
                if not turn_gt:
                    turn_results.append({
                        "turn": turn_idx, "predicted": [],
                        "expected": [], "success": True, "score": 1.0,
                    })
                    continue
                conversation.append({
                    "role": "user",
                    "content": (
                        "The previously unavailable function(s) are now available. "
                        "Please execute any pending operations from earlier requests."
                    ),
                })
            else:
                for msg in turn_messages:
                    conversation.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", ""),
                    })

            # Call the LLM with native tool calling
            try:
                response = self.llm.invoke_with_tools(
                    list(conversation), tools,
                )
            except Exception as e:
                turn_results.append({
                    "turn": turn_idx, "predicted": [],
                    "expected": turn_gt, "success": False,
                    "score": 0.0, "error": str(e),
                })
                all_correct = False
                conversation.append({"role": "assistant", "content": ""})
                continue

            # Extract tool calls from the response
            message = response.choices[0].message
            tool_calls = message.tool_calls

            if tool_calls:
                # Parse each tool call into the standard dict format
                predicted_calls = []
                for tc in tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    predicted_calls.append({
                        "name": tc.function.name,
                        "arguments": args,
                    })

                # Append assistant message with tool_calls to conversation
                conversation.append(message.model_dump())

                # Append simulated tool results for each call
                for tc in tool_calls:
                    conversation.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({
                            "status": "success",
                            "result": "Operation completed.",
                        }),
                    })
            else:
                # Model returned text, no tool calls
                predicted_calls = []
                content = message.content or ""
                conversation.append({"role": "assistant", "content": content})

            # Compare against this turn's ground truth
            if turn_gt:
                turn_success, turn_score = self._evaluate_string_format(
                    predicted_calls, turn_gt,
                    func_params_lookup=func_params_lookup,
                )
            else:
                turn_success = len(predicted_calls) == 0
                turn_score = 1.0 if turn_success else 0.0

            turn_results.append({
                "turn": turn_idx,
                "predicted": predicted_calls,
                "expected": turn_gt,
                "success": turn_success,
                "score": turn_score,
            })
            if not turn_success:
                all_correct = False

        execution_time = time.time() - start_time
        total_score = (
            sum(r["score"] for r in turn_results) / len(turn_results)
            if turn_results else 0.0
        )

        # Log per-sample multi-turn summary (non-TTY only to avoid clobbering progress bar)
        if not sys.stdout.isatty():
            sample_id = sample.get("id", "?")
            turn_marks = "".join(
                "+" if r["success"] else "-" for r in turn_results
            )
            print(f"      [{sample_id}] turns: [{turn_marks}] score: {total_score:.2f}")

        return {
            "success": all_correct,
            "score": total_score,
            "predicted": [r["predicted"] for r in turn_results],
            "expected": ground_truth,
            "turn_results": turn_results,
            "execution_time": execution_time,
            "sample_id": sample.get("id", ""),
            "category": self.category or sample.get("category", "unknown"),
        }

    def _load_multi_turn_functions(self, sample: Dict[str, Any]) -> List[Dict]:
        """Load function definitions for a multi-turn sample.

        Reads JSONL files from ``data/BFCL/multi_turn_func_doc/`` based on the
        sample's ``involved_classes``, filtering out ``excluded_function`` entries.
        """
        involved_classes = sample.get("involved_classes", [])
        excluded = set(sample.get("excluded_function", []))

        func_doc_dir = self.dataset.bfcl_data_dir / "multi_turn_func_doc"

        functions: list[dict] = []
        for cls_name in involved_classes:
            doc_file = self._CLASS_TO_DOC_FILE.get(cls_name)
            if not doc_file:
                logger.warning("No doc file mapping for class: %s", cls_name)
                continue
            doc_path = func_doc_dir / doc_file
            if not doc_path.exists():
                logger.warning("Doc file not found: %s", doc_path)
                continue

            with open(doc_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        func_def = json.loads(line)
                        func_name = func_def.get("name", "")
                        if func_name not in excluded:
                            functions.append(func_def)
                    except json.JSONDecodeError:
                        continue

        return functions

    def _build_multi_turn_system_prompt(self, functions: List[Dict]) -> str:
        """Build a system prompt listing available functions for multi-turn evaluation."""
        if not functions:
            return ""

        prompt = (
            "You are a function-calling assistant. For each user request, "
            "output ONLY a valid JSON array of function calls to execute. "
            "No explanation, no markdown fences, no extra text.\n\n"
            "Output format:\n"
            '[{"name": "<function_name>", "arguments": {"<param>": "<value>"}}, ...]\n\n'
            "If no function call is needed, output: []\n\n"
            "## Available Functions\n\n"
        )

        for func in functions:
            func_name = func.get("name", "")
            func_desc = func.get("description", "")
            func_params = func.get("parameters", {})

            prompt += f"### {func_name}\n"
            if func_desc:
                prompt += f"{func_desc}\n"
            if func_params:
                properties = func_params.get("properties", {})
                required = func_params.get("required", [])
                if properties:
                    prompt += "Parameters:\n"
                    for pname, pinfo in properties.items():
                        req_marker = " (required)" if pname in required else ""
                        ptype = pinfo.get("type", "any")
                        pdesc = pinfo.get("description", "")
                        prompt += f"  - {pname} ({ptype}{req_marker}): {pdesc}\n"
            prompt += "\n"

        return prompt

    def _create_empty_results(self, agent: Any) -> Dict[str, Any]:
        """Create empty evaluation results."""
        return {
            "benchmark": "BFCL",
            "agent_name": getattr(agent, 'name', 'Unknown'),
            "evaluation_mode": self.evaluation_mode,
            "category": self.category,
            "total_samples": 0,
            "correct_samples": 0,
            "overall_accuracy": 0.0,
            "category_metrics": {},
            "detailed_results": []
        }

    def _build_function_calling_prompt(self, question: str, functions: List[Dict]) -> str:
        """Build a BFCL-specific function-calling prompt.

        The prompt is designed to elicit a strict JSON array response so that
        downstream extraction is reliable.
        """
        if not functions:
            return question

        prompt = (
            "You are a function-calling assistant. Your job is to select the "
            "correct function(s) and provide the correct arguments based on the "
            "user's question.\n\n"
            "## Available Functions\n\n"
        )

        for i, func in enumerate(functions, 1):
            func_name = func.get("name", f"function_{i}")
            func_desc = func.get("description", "")
            func_params = func.get("parameters", {})

            prompt += f"### {func_name}\n"
            if func_desc:
                prompt += f"{func_desc}\n"
            if func_params:
                prompt += f"Parameters:\n```json\n{json.dumps(func_params, ensure_ascii=False, indent=2)}\n```\n"
            prompt += "\n"

        prompt += (
            "## Output Format\n\n"
            "Return ONLY a valid JSON array of function calls. "
            "No explanation, no markdown fences, no extra text.\n\n"
            "Example:\n"
            '[{"name": "<function_name>", "arguments": {"<param>": "<value>"}}, ...]\n\n'
            f"## User Question\n\n{question}"
        )

        return prompt

    def _extract_function_calls(self, response: str) -> List[Dict[str, Any]]:
        """Extract function calls from the LLM response.

        Handles common response formats:
        - Clean JSON array: ``[{...}, {...}]``
        - Bare comma-separated objects: ``{...}, {...}``
        - JSON embedded in markdown fences or surrounding text
        """
        if not response or not isinstance(response, str):
            return []

        text = response.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            # Remove opening fence (```json or ```)
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1:]
            # Remove closing fence
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3].rstrip()

        text = text.strip()

        # 1. Direct JSON array
        if text.startswith("["):
            try:
                result = json.loads(text)
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        # 2. Bare object(s) — wrap with brackets: "{...}, {...}" -> "[{...}, {...}]"
        if text.startswith("{"):
            try:
                result = json.loads(f"[{text}]")
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        # 3. Bracket-depth scan: find the first top-level JSON array in mixed text
        calls = self._scan_for_json_array(text)
        if calls is not None:
            return calls

        # 4. Bracket-depth scan for individual objects
        calls = self._scan_for_json_objects(text)
        if calls:
            return calls

        return []

    def _scan_for_json_array(self, text: str) -> Optional[List[Dict]]:
        """Find the first balanced ``[...]`` substring that looks like function calls.

        Skips arrays that don't contain at least one dict with a ``"name"`` key,
        so that stray arrays in reasoning text (e.g. ``["Tenet", "No Time To Die"]``)
        are not mistakenly returned.
        """
        pos = 0
        while pos < len(text):
            start = text.find("[", pos)
            if start == -1:
                return None

            depth = 0
            in_string = False
            escape_next = False
            end = -1

            for i in range(start, len(text)):
                ch = text[i]

                if escape_next:
                    escape_next = False
                    continue

                if ch == "\\":
                    if in_string:
                        escape_next = True
                    continue

                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if in_string:
                    continue

                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break

            if end == -1:
                return None

            candidate = text[start:end + 1]
            try:
                result = json.loads(candidate)
                if isinstance(result, list) and any(
                    isinstance(item, dict) and "name" in item for item in result
                ):
                    return result
            except json.JSONDecodeError:
                pass

            # This array wasn't a function-call array; keep scanning
            pos = end + 1

        return None

    def _scan_for_json_objects(self, text: str) -> List[Dict]:
        """Find all balanced ``{...}`` substrings that contain a ``name`` key."""
        objects = []
        i = 0
        while i < len(text):
            if text[i] != "{":
                i += 1
                continue

            depth = 0
            in_string = False
            escape_next = False
            start = i

            for j in range(start, len(text)):
                ch = text[j]

                if escape_next:
                    escape_next = False
                    continue

                if ch == "\\":
                    if in_string:
                        escape_next = True
                    continue

                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if in_string:
                    continue

                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:j + 1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict) and "name" in obj:
                                objects.append(obj)
                        except json.JSONDecodeError:
                            pass
                        i = j + 1
                        break
            else:
                # Unbalanced braces — skip this '{'
                i = start + 1

        return objects

    def _evaluate_ast_matching(self, predicted: List[Dict], expected: List) -> tuple[bool, float]:
        """AST matching evaluation.

        Supports two ground truth formats:
        1. BFCL v4 format: [{"func_name": {"param": [value1, value2]}}]
        2. String format: ["func_name(param=value)"]
        """
        if not expected:
            return len(predicted) == 0, 1.0 if len(predicted) == 0 else 0.0

        try:
            # Detect ground truth format
            if expected and isinstance(expected[0], dict):
                # BFCL v4 format
                return self._evaluate_bfcl_v4_format(predicted, expected)
            else:
                # String format (legacy)
                return self._evaluate_string_format(predicted, expected)

        except Exception as e:
            print(f"   ⚠️ Evaluation error: {e}")
            return False, 0.0

    def _evaluate_bfcl_v4_format(self, predicted: List[Dict], expected: List[Dict]) -> tuple[bool, float]:
        """Evaluate BFCL v4 format ground truth.

        BFCL v4 format:
        predicted: [{"name": "func_name", "arguments": {"param": value}}]
        expected: [{"func_name": {"param": [value1, value2]}}]
        """
        if len(predicted) != len(expected):
            return False, 0.0

        matches = 0
        for pred_call in predicted:
            if not isinstance(pred_call, dict) or "name" not in pred_call:
                continue

            pred_func_name = pred_call["name"]
            pred_args = pred_call.get("arguments", {})

            # Find matching function call in expected
            for exp_call in expected:
                if not isinstance(exp_call, dict):
                    continue

                # expected format: {"func_name": {"param": [values]}}
                for exp_func_name, exp_params in exp_call.items():
                    if exp_func_name != pred_func_name:
                        continue

                    # Compare parameters
                    if self._compare_parameters(pred_args, exp_params):
                        matches += 1
                        break

        success = matches == len(expected)
        score = matches / len(expected) if expected else 0.0
        return success, score

    def _compare_parameters(self, pred_params: Dict, exp_params: Dict) -> bool:
        """Compare predicted parameters with expected parameters.

        Args:
            pred_params: {"param": value}
            exp_params: {"param": [value1, value2]}  # Array represents multiple acceptable values
        """
        # Check all required parameters
        for param_name, expected_values in exp_params.items():
            if param_name not in pred_params:
                # Parameter missing, check if empty string is a default value
                if not isinstance(expected_values, list) or "" not in expected_values:
                    return False
                continue

            pred_value = pred_params[param_name]

            # Normalize caret to double-star in string values (e.g. "3x^2" -> "3x**2")
            if isinstance(pred_value, str):
                pred_value = pred_value.replace("^", "**")

            # expected_values is an array containing all acceptable values
            if isinstance(expected_values, list):
                # Check if pred_value is in the list of acceptable values
                if pred_value not in expected_values:
                    # Try type-converted comparison
                    if str(pred_value) not in [str(v) for v in expected_values]:
                        return False
            else:
                # Single value comparison
                if pred_value != expected_values and str(pred_value) != str(expected_values):
                    return False

        return True

    def _evaluate_string_format(
        self,
        predicted: List[Dict],
        expected: List[str],
        func_params_lookup: Optional[Dict[str, List[str]]] = None,
    ) -> tuple[bool, float]:
        """Evaluate string format ground truth (legacy)."""
        if func_params_lookup is None:
            func_params_lookup = {}

        # Convert predicted results to string form
        predicted_strs = []
        for call in predicted:
            if isinstance(call, dict) and "name" in call:
                func_name = call["name"]
                args = call.get("arguments", {})
                # Build function call string
                if args:
                    args_str = ", ".join([f"{k}={repr(v)}" for k, v in args.items()])
                    call_str = f"{func_name}({args_str})"
                else:
                    call_str = f"{func_name}()"
                predicted_strs.append(call_str)

        # Simple string matching evaluation
        if len(predicted_strs) != len(expected):
            return False, 0.0

        # Check if each function call matches
        matches = 0
        for pred_str in predicted_strs:
            for exp_str in expected:
                if self._ast_strings_match(pred_str, exp_str, func_params_lookup):
                    matches += 1
                    break

        success = matches == len(expected)
        score = matches / len(expected) if expected else 0.0

        return success, score

    def _ast_strings_match(
        self,
        pred: str,
        expected: str,
        func_params_lookup: Optional[Dict[str, List[str]]] = None,
    ) -> bool:
        """Compare whether two function call strings match at the AST level.

        When *func_params_lookup* is supplied, both strings are first
        normalised so that positional args become keyword args.
        """
        if func_params_lookup is None:
            func_params_lookup = {}

        if func_params_lookup:
            pred = self._normalize_call_to_keyword_args(pred, func_params_lookup)
            expected = self._normalize_call_to_keyword_args(expected, func_params_lookup)

        try:
            pred_ast = ast.parse(pred, mode='eval')
            exp_ast = ast.parse(expected, mode='eval')
            return ast.dump(pred_ast) == ast.dump(exp_ast)
        except Exception:
            return pred.strip() == expected.strip()

    def _evaluate_execution(self, predicted: List[Dict], expected: List[str], functions: List[Dict]) -> tuple[bool, float]:
        """Execution evaluation (simplified version)."""
        # Simplified execution evaluation
        # In practice, a secure code execution environment is needed
        return self._evaluate_ast_matching(predicted, expected)

    def export_to_bfcl_format(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path],
        include_inference_log: bool = True
    ) -> None:
        """Export evaluation results in BFCL official format.

        BFCL official format example:
        {
            "id": "simple_python_0",
            "model_result": [
                {
                    "name": "calculate_triangle_area",
                    "arguments": {"base": 10, "height": 5, "unit": "units"}
                }
            ],
            "inference_log": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        }

        Args:
            results: Evaluation results returned by the evaluate() method.
            output_path: Output file path.
            include_inference_log: Whether to include the inference log.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to BFCL format
        bfcl_results = []

        for detail in results.get("detailed_results", []):
            # Convert predicted to string-format function calls
            predicted = detail.get("predicted", [])
            result_string = ""

            if predicted:
                call = predicted[0]  # Usually only one function call
                if isinstance(call, dict) and "name" in call:
                    func_name = call["name"]
                    args = call.get("arguments", {})

                    # Build function call string
                    if args:
                        args_str = ", ".join([f"{k}={repr(v)}" for k, v in args.items()])
                        result_string = f"{func_name}({args_str})"
                    else:
                        result_string = f"{func_name}()"

            bfcl_item = {
                "id": detail.get("sample_id", ""),
                "result": result_string  # BFCL expects a single string
            }

            # Add inference log (if needed)
            if include_inference_log:
                question = detail.get("question", "")
                response = detail.get("response", "")

                bfcl_item["inference_log"] = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response}
                ]

            bfcl_results.append(bfcl_item)

        # Write in JSONL format (one JSON object per line)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in bfcl_results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

