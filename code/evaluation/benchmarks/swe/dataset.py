"""
SWE-bench Dataset Loading Module

Loads the SWE-bench (Software Engineering Benchmark) dataset,
used to evaluate an agent's ability to fix real GitHub issues.
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json


class SWEDataset:
    """SWE-bench dataset loader

    Loads SWE-bench data from local JSONL files.

    SWE-bench is a software engineering evaluation benchmark where each instance
    contains a GitHub issue (problem_statement), the corresponding repository state
    (repo + base_commit), and test lists for verification (FAIL_TO_PASS / PASS_TO_PASS).

    Attributes:
        split: Dataset split (dev/test/train)
        data_dir: Data directory path
        repo_filter: Optional repository name filter
        data: Loaded data list
    """

    def __init__(
        self,
        split: str = "dev",
        data_dir: Optional[Union[str, Path]] = None,
        repo_filter: Optional[str] = None,
    ):
        """Initialize the SWE-bench dataset loader.

        Args:
            split: Dataset split (dev/test/train)
            data_dir: Directory containing JSONL data files, defaults to data/SWE
            repo_filter: Only keep instances from the specified repo (e.g. "astropy/astropy")
        """
        self.split = split
        self.data_dir = Path(data_dir) if data_dir else Path("data/SWE")
        self.repo_filter = repo_filter
        self.data: List[Dict[str, Any]] = []

    def load(self) -> List[Dict[str, Any]]:
        """Load the dataset.

        Returns:
            List of standardized SWE-bench instances
        """
        jsonl_path = self.data_dir / f"{self.split}.jsonl"

        if not jsonl_path.exists():
            print(f"   [Warning] Data file not found: {jsonl_path}")
            return []

        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                item = self._standardize_item(raw)
                self.data.append(item)

        # Filter by repository
        if self.repo_filter:
            self.data = [
                item for item in self.data if item.get("repo") == self.repo_filter
            ]

        print(f"[Done] SWE-bench dataset loaded")
        print(f"   Split       : {self.split}")
        print(f"   Repo filter : {self.repo_filter or 'all'}")
        print(f"   Samples     : {len(self.data)}")

        return self.data

    def _standardize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize the data item format.

        Maps raw JSONL fields to a unified schema.
        """
        fail_to_pass = item.get("FAIL_TO_PASS", "[]")
        pass_to_pass = item.get("PASS_TO_PASS", "[]")

        # FAIL_TO_PASS / PASS_TO_PASS may be JSON strings in the JSONL
        if isinstance(fail_to_pass, str):
            try:
                fail_to_pass = json.loads(fail_to_pass)
            except json.JSONDecodeError:
                fail_to_pass = []
        if isinstance(pass_to_pass, str):
            try:
                pass_to_pass = json.loads(pass_to_pass)
            except json.JSONDecodeError:
                pass_to_pass = []

        return {
            "instance_id": item.get("instance_id", ""),
            "repo": item.get("repo", ""),
            "base_commit": item.get("base_commit", ""),
            "problem_statement": item.get("problem_statement", ""),
            "hints_text": item.get("hints_text", ""),
            "patch": item.get("patch", ""),
            "test_patch": item.get("test_patch", ""),
            "FAIL_TO_PASS": fail_to_pass,
            "PASS_TO_PASS": pass_to_pass,
            "version": item.get("version", ""),
            "created_at": item.get("created_at", ""),
            "environment_setup_commit": item.get("environment_setup_commit", ""),
            "raw_item": item,
        }

    def get_sample(self, index: int) -> Dict[str, Any]:
        """Get a single sample.

        Args:
            index: Sample index

        Returns:
            Sample data
        """
        if not self.data:
            self.load()
        return self.data[index] if index < len(self.data) else {}

    def get_by_repo(self, repo: str) -> List[Dict[str, Any]]:
        """Get all instances for a specific repository.

        Args:
            repo: Full repository name (e.g. "django/django")

        Returns:
            All samples from that repository
        """
        if not self.data:
            self.load()
        return [item for item in self.data if item.get("repo") == repo]

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics.

        Returns:
            Statistics dictionary
        """
        if not self.data:
            self.load()

        # Count by repository
        repo_counts: Dict[str, int] = {}
        for item in self.data:
            repo = item.get("repo", "unknown")
            repo_counts[repo] = repo_counts.get(repo, 0) + 1

        # Count by version
        version_counts: Dict[str, int] = {}
        for item in self.data:
            version = item.get("version", "unknown")
            version_counts[version] = version_counts.get(version, 0) + 1

        return {
            "total_samples": len(self.data),
            "split": self.split,
            "num_repos": len(repo_counts),
            "repo_distribution": repo_counts,
            "version_distribution": version_counts,
        }

    def __len__(self) -> int:
        """Return the dataset size."""
        if not self.data:
            self.load()
        return len(self.data)

    def __bool__(self) -> bool:
        """Dataset object is always truthy (prevents falsy on empty data)."""
        return True

    def __iter__(self):
        """Iterator."""
        if not self.data:
            self.load()
        return iter(self.data)
