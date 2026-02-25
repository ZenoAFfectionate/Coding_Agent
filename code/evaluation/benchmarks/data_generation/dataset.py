"""
AIME Dataset Loader

Loads AIME math problem datasets, supporting:
- Loading real problems from HuggingFace
- Loading generated problem data
- Unified data formatting
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from huggingface_hub import snapshot_download


class AIDataset:
    """AIME dataset loader"""

    def __init__(
        self,
        dataset_type: str = "generated",  # "generated" or "real"
        data_path: Optional[str] = None,
        year: Optional[int] = None,  # For real problems, e.g. 2024, 2025
        cache_dir: Optional[str] = None
    ):
        """
        Initialize AIME dataset.

        Args:
            dataset_type: Dataset type, "generated" or "real"
            data_path: Local data path (for generated type)
            year: AIME year (for real type), e.g. 2024, 2025
            cache_dir: Cache directory
        """
        self.dataset_type = dataset_type
        self.data_path = data_path
        self.year = year
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/hello_agents/aime")

        self.problems: List[Dict[str, Any]] = []

    def load(self) -> List[Dict[str, Any]]:
        """
        Load the dataset.

        Returns:
            List of problems, each containing:
            - problem_id: Problem ID
            - problem: Problem description
            - answer: Answer
            - solution: Solution process (optional)
            - difficulty: Difficulty level (optional)
            - topic: Topic (optional)
        """
        if self.dataset_type == "generated":
            return self._load_generated_data()
        elif self.dataset_type == "real":
            return self._load_real_data()
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")

    def _load_generated_data(self) -> List[Dict[str, Any]]:
        """Load generated data."""
        if not self.data_path:
            raise ValueError("data_path is required for generated dataset")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        print(f"[Load] Loading generated data: {self.data_path}")

        with open(self.data_path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)

            if first_char == '[':
                # Standard JSON array
                data = json.load(f)
            else:
                # JSONL format (one JSON object per line)
                data = []
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))

        # Unify data format
        problems = []
        for idx, item in enumerate(data):
            problem = {
                "problem_id": item.get("id", f"gen_{idx}"),
                "problem": item.get("problem", item.get("question", "")),
                "answer": item.get("answer", ""),
                "solution": item.get("solution", item.get("reasoning", "")),
                "difficulty": item.get("difficulty", None),
                "topic": item.get("topic", item.get("category", None))
            }
            problems.append(problem)

        self.problems = problems
        print(f"[Done] Loaded {len(problems)} generated problems")
        return problems

    def _load_real_data(self) -> List[Dict[str, Any]]:
        """Load real AIME problems from HuggingFace."""
        if not self.year:
            raise ValueError("year is required for real dataset")

        print(f"[Load] Loading AIME {self.year} real problems from HuggingFace...")

        try:
            # Use AIME 2025 dataset
            repo_id = "math-ai/aime25"

            print(f"   Dataset: {repo_id}")

            # Check common local paths before downloading
            local_candidates = [
                Path(f"data/AIME/test.jsonl"),
                Path(f"data/aime25/test.jsonl"),
                Path(f"data/aime/test.jsonl"),
            ]

            local_dir = None
            data_file = None
            for candidate in local_candidates:
                if candidate.exists():
                    data_file = candidate
                    print(f"   Found local data file: {data_file}")
                    break

            if data_file is None:
                # Download files using snapshot_download
                local_dir = snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    cache_dir=self.cache_dir
                )

                # Find JSONL data file
                data_files = list(Path(local_dir).glob("*.jsonl"))

                if not data_files:
                    raise FileNotFoundError(f"No JSONL data file found in {repo_id}")

                data_file = data_files[0]
                print(f"   Found data file: {data_file.name}")

            # Load JSONL data
            data = []
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))

            # Unify data format (AIME 2025 uses lowercase field names)
            problems = []
            for idx, item in enumerate(data):
                problem = {
                    "problem_id": item.get("id", f"aime_2025_{idx}"),
                    "problem": item.get("problem", ""),
                    "answer": item.get("answer", ""),
                    "solution": item.get("solution", ""),  # AIME 2025 may not have solution field
                    "difficulty": item.get("difficulty", None),
                    "topic": item.get("topic", None)
                }
                problems.append(problem)

            self.problems = problems
            print(f"[Done] Loaded {len(problems)} AIME {self.year} real problems")
            return problems

        except Exception as e:
            print(f"[Error] Failed to load: {e}")
            print(f"Hint: Make sure huggingface_hub is installed and HF_TOKEN is configured")
            raise

    def get_problem(self, problem_id: str) -> Optional[Dict[str, Any]]:
        """Get a problem by its ID."""
        for problem in self.problems:
            if problem["problem_id"] == problem_id:
                return problem
        return None

    def get_problems_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Get problems by topic."""
        return [p for p in self.problems if p.get("topic") == topic]

    def get_problems_by_difficulty(self, min_diff: int, max_diff: int) -> List[Dict[str, Any]]:
        """Get problems within a difficulty range."""
        return [
            p for p in self.problems
            if p.get("difficulty") and min_diff <= p["difficulty"] <= max_diff
        ]

    def __len__(self) -> int:
        """Return the dataset size."""
        return len(self.problems)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Support index-based access."""
        return self.problems[idx]
