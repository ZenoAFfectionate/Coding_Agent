"""
TritonBench Dataset Loading Module

Loads TritonBench G-channel and T-channel data for evaluating
Triton GPU kernel generation.
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import ast
import json
import logging

logger = logging.getLogger(__name__)


# Separator used in reference .py files to split gold code from test code
SEPARATOR = "#" * 146


class TritonBenchDataset:
    """TritonBench dataset loader

    Loads TritonBench data from local JSON/JSONL files and corresponding
    reference Python files.

    Attributes:
        channel: Evaluation channel ("G" or "T")
        data_dir: Data directory path
        instruction_mode: Instruction type for G channel ("simple" or "complex")
        difficulty: Optional difficulty filter (1-5)
        data: Loaded data list
    """

    def __init__(
        self,
        channel: str = "G",
        data_dir: Optional[Union[str, Path]] = None,
        instruction_mode: str = "simple",
        difficulty: Optional[int] = None,
    ):
        """Initialize the TritonBench dataset loader.

        Args:
            channel: "G" for GitHub-sourced kernels, "T" for PyTorch-to-Triton
            data_dir: Directory containing TritonBench data files
            instruction_mode: "simple" or "complex" (G channel only; T always uses full spec)
            difficulty: Filter by difficulty level (1-5), None for all
        """
        if channel not in ("G", "T"):
            raise ValueError(f"Invalid channel: {channel!r}. Must be 'G' or 'T'.")
        self.channel = channel
        self.data_dir = Path(data_dir) if data_dir else Path("data/TRIB")
        self.instruction_mode = instruction_mode
        self.difficulty = difficulty
        self.data: List[Dict[str, Any]] = []

    def load(self) -> List[Dict[str, Any]]:
        """Load the dataset.

        Returns:
            List of standardized task instances
        """
        if self.channel == "G":
            self.data = self._load_g_channel()
        else:
            self.data = self._load_t_channel()

        # Filter by difficulty
        if self.difficulty is not None:
            self.data = [
                item for item in self.data
                if item.get("difficulty") == self.difficulty
            ]

        print(f"[Done] TritonBench dataset loaded")
        print(f"   Channel          : {self.channel}")
        print(f"   Instruction mode : {self.instruction_mode}")
        print(f"   Difficulty filter: {self.difficulty or 'all'}")
        print(f"   Samples          : {len(self.data)}")

        return self.data

    def _load_g_channel(self) -> List[Dict[str, Any]]:
        """Load G-channel data from TritonBench_G_v1.json and reference .py files."""
        json_path = self.data_dir / "TritonBench_G_v1.json"
        ref_dir = self.data_dir / "TritonBench_G_v1"

        if not json_path.exists():
            print(f"   [Warning] Data file not found: {json_path}")
            return []

        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        items = []
        for raw in raw_data:
            filename = raw.get("file", "")
            task_id = Path(filename).stem

            # Choose instruction based on mode
            if self.instruction_mode == "complex":
                instruction = raw.get("comp_instru", "")
            else:
                instruction = raw.get("simp_instru", "")

            # Gold code comes from the JSON 'output' field
            gold_code = raw.get("output", "")

            # Extract test code from reference .py file
            test_code = ""
            ref_file = ref_dir / filename
            if ref_file.exists():
                ref_content = ref_file.read_text(encoding="utf-8")
                parts = ref_content.split(SEPARATOR)
                if len(parts) >= 2:
                    test_code = parts[1].strip()

            # Difficulty may be stored as string in JSON
            raw_diff = raw.get("difficulty")
            difficulty = int(raw_diff) if raw_diff is not None else None

            items.append({
                "task_id": task_id,
                "channel": "G",
                "instruction": instruction.strip(),
                "gold_code": gold_code,
                "api_spec": self._extract_api_spec(gold_code),
                "test_code": test_code,
                "reference_file": filename,
                "repo": raw.get("repo", ""),
                "difficulty": difficulty,
                "star": raw.get("star"),
            })

        return items

    def _load_t_channel(self) -> List[Dict[str, Any]]:
        """Load T-channel data from TritonBench_T_v1.jsonl and reference .py files."""
        jsonl_path = self.data_dir / "TritonBench_T_v1.jsonl"
        ref_dir = self.data_dir / "TritonBench_T_v1"

        if not jsonl_path.exists():
            print(f"   [Warning] Data file not found: {jsonl_path}")
            return []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            content = f.read()
        raw_data = json.loads(content)

        items = []
        for raw in raw_data:
            name = raw.get("name", "")
            filename = raw.get("file", f"{name}.py")
            task_id = Path(filename).stem if filename else name

            # Build instruction from structured fields
            instruction = self._build_t_instruction(raw)

            # Extract gold code and test code from reference .py file
            gold_code = ""
            test_code = ""
            ref_file = ref_dir / filename
            if ref_file.exists():
                ref_content = ref_file.read_text(encoding="utf-8")
                parts = ref_content.split(SEPARATOR)
                if len(parts) >= 2:
                    gold_code = parts[0].strip()
                    test_code = parts[1].strip()

            # Difficulty may be stored as string in JSON
            raw_diff = raw.get("difficulty")
            difficulty = int(raw_diff) if raw_diff is not None else None

            items.append({
                "task_id": task_id,
                "channel": "T",
                "instruction": instruction,
                "gold_code": gold_code,
                "api_spec": self._extract_api_spec(gold_code),
                "test_code": test_code,
                "reference_file": filename,
                "torch_code": raw.get("torch_code", ""),
                "func_inputs": raw.get("func_inputs", ""),
                "description": raw.get("description", ""),
                "math": raw.get("math", ""),
                "example": raw.get("example", ""),
                "difficulty": difficulty,
            })

        return items

    def _build_t_instruction(self, raw: Dict[str, Any]) -> str:
        """Build a natural-language instruction for a T-channel task."""
        parts = []

        desc = raw.get("description", "")
        if desc:
            parts.append(f"Description: {desc}")

        func_inputs = raw.get("func_inputs", "")
        if func_inputs:
            parts.append(f"Function signature: {func_inputs}")

        math = raw.get("math", "")
        if math:
            parts.append(f"Mathematical formula: {math}")

        example = raw.get("example", "")
        if example:
            parts.append(f"Example:\n{example}")

        torch_code = raw.get("torch_code", "")
        if torch_code:
            parts.append(f"PyTorch reference implementation:\n{torch_code}")

        return "\n\n".join(parts)

    @staticmethod
    def _extract_api_spec(gold_code: str) -> str:
        """Extract public API signatures from gold code via AST parsing.

        Produces a specification telling the model exactly which kernel functions,
        wrapper functions, classes, and module-level aliases it must export.
        This bridges the information gap: the model learns *what* to implement
        (names + signatures) without seeing *how* (the implementation body).

        Returns an empty string if parsing fails or no API is found.
        """
        if not gold_code or not gold_code.strip():
            return ""

        try:
            tree = ast.parse(gold_code)
        except SyntaxError:
            logger.debug("Failed to parse gold code for API extraction")
            return ""

        lines: List[str] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if decorated with @triton.jit or @triton.autotune
                is_kernel = any(
                    "triton" in ast.unparse(d) for d in node.decorator_list
                )
                params = ast.unparse(node.args)
                if is_kernel:
                    lines.append(f"- Kernel function: `{node.name}({params})`")
                else:
                    lines.append(f"- Wrapper function: `{node.name}({params})`")

            elif isinstance(node, ast.ClassDef):
                bases = ", ".join(ast.unparse(b) for b in node.bases)
                lines.append(f"- Class: `{node.name}({bases})`")
                for member in node.body:
                    if isinstance(member, ast.FunctionDef):
                        mparams = ast.unparse(member.args)
                        lines.append(f"  - Method: `{member.name}({mparams})`")

            elif isinstance(node, ast.Assign):
                # Module-level aliases like: fn = ClassName.apply
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        value_src = ast.unparse(node.value)
                        lines.append(f"- Module-level alias: `{target.id} = {value_src}`")

        return "\n".join(lines)

    def get_sample(self, index: int) -> Dict[str, Any]:
        """Get a single sample by index."""
        if not self.data:
            self.load()
        return self.data[index] if index < len(self.data) else {}

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.data:
            self.load()

        difficulty_counts: Dict[int, int] = {}
        for item in self.data:
            d = item.get("difficulty")
            if d is not None:
                difficulty_counts[d] = difficulty_counts.get(d, 0) + 1

        repo_counts: Dict[str, int] = {}
        for item in self.data:
            repo = item.get("repo", "")
            if repo:
                repo_counts[repo] = repo_counts.get(repo, 0) + 1

        return {
            "total_samples": len(self.data),
            "channel": self.channel,
            "difficulty_distribution": difficulty_counts,
            "repo_distribution": repo_counts,
        }

    def __len__(self) -> int:
        if not self.data:
            self.load()
        return len(self.data)

    def __bool__(self) -> bool:
        return True

    def __iter__(self):
        if not self.data:
            self.load()
        return iter(self.data)
