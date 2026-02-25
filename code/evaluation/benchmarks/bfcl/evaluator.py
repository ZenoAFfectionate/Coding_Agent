"""
BFCL Evaluator Module

Responsible for evaluating the agent's performance on the BFCL benchmark.
"""

from typing import Dict, Any, List, Optional, Union
import json
import ast
import logging
import time
from pathlib import Path
from code.evaluation.benchmarks.bfcl.dataset import BFCLDataset
from code.evaluation.benchmarks.bfcl.metrics import BFCLMetrics

logger = logging.getLogger(__name__)

# Evaluation prompts directory (benchmark-specific prompts)
EVAL_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


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

        print(f"   Sample count: {len(dataset)}")

        # Run evaluation
        results = []
        categories = {}
        total = len(dataset)
        bar_width = 30

        for i, sample in enumerate(dataset):
            # Print progress bar
            done = i + 1
            filled = int(bar_width * done / total)
            bar = "█" * filled + "░" * (bar_width - filled)
            pct = done / total * 100
            correct_so_far = sum(1 for r in results if r.get("success"))
            print(f"\r   [{bar}] {done}/{total} ({pct:.0f}%) | ✓ {correct_so_far}", end="", flush=True)

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
        try:
            # Prepare input
            question = sample.get("question", "")
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

    def _evaluate_string_format(self, predicted: List[Dict], expected: List[str]) -> tuple[bool, float]:
        """Evaluate string format ground truth (legacy)."""
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
                if self._ast_strings_match(pred_str, exp_str):
                    matches += 1
                    break

        success = matches == len(expected)
        score = matches / len(expected) if expected else 0.0

        return success, score

    def _ast_strings_match(self, pred: str, expected: str) -> bool:
        """Compare whether two function call strings match at the AST level."""
        try:
            # Try to parse as AST and compare
            pred_ast = ast.parse(pred, mode='eval')
            exp_ast = ast.parse(expected, mode='eval')
            return ast.dump(pred_ast) == ast.dump(exp_ast)
        except:
            # If AST parsing fails, use string comparison
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

