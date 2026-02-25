"""
BFCL Dataset Loading Module

Responsible for loading the Berkeley Function Calling Leaderboard dataset.
Supports loading from the BFCL official data directory, including test data and ground truth.
"""

from typing import List, Dict, Any, Optional, Union
import json
import os
from pathlib import Path


class BFCLDataset:
    """BFCL Dataset Loader

    Supports loading datasets from the BFCL official data directory,
    including test data and ground truth.

    Dataset categories (BFCL v4):
    - simple_python: Simple Python function calls
    - simple_java: Simple Java function calls
    - simple_javascript: Simple JavaScript function calls
    - multiple: Multiple function calls
    - parallel: Parallel function calls
    - parallel_multiple: Parallel multiple function calls
    - irrelevance: Irrelevance detection
    - live_simple: User-contributed simple function calls
    - live_multiple: User-contributed multiple function calls
    - live_parallel: User-contributed parallel function calls
    - multi_turn_base: Multi-turn conversation base
    - multi_turn_miss_func: Multi-turn conversation with missing functions
    - multi_turn_miss_param: Multi-turn conversation with missing parameters
    - multi_turn_long_context: Multi-turn conversation with long context

    Attributes:
        bfcl_data_dir: BFCL official data directory path
        category: Evaluation category
        data: List of loaded test data
        ground_truth: Ground truth dictionary, keyed by sample ID
    """

    # Standard category mapping for BFCL v4 dataset
    CATEGORY_MAPPING = {
        "simple_python": "BFCL_v4_simple_python",
        "simple_java": "BFCL_v4_simple_java",
        "simple_javascript": "BFCL_v4_simple_javascript",
        "multiple": "BFCL_v4_multiple",
        "parallel": "BFCL_v4_parallel",
        "parallel_multiple": "BFCL_v4_parallel_multiple",
        "irrelevance": "BFCL_v4_irrelevance",
        "live_simple": "BFCL_v4_live_simple",
        "live_multiple": "BFCL_v4_live_multiple",
        "live_parallel": "BFCL_v4_live_parallel",
        "live_parallel_multiple": "BFCL_v4_live_parallel_multiple",
        "live_irrelevance": "BFCL_v4_live_irrelevance",
        "live_relevance": "BFCL_v4_live_relevance",
        "multi_turn_base": "BFCL_v4_multi_turn_base",
        "multi_turn_miss_func": "BFCL_v4_multi_turn_miss_func",
        "multi_turn_miss_param": "BFCL_v4_multi_turn_miss_param",
        "multi_turn_long_context": "BFCL_v4_multi_turn_long_context",
        "memory": "BFCL_v4_memory",
        "web_search": "BFCL_v4_web_search",
    }

    def __init__(
        self,
        bfcl_data_dir: Union[str, Path] = "data/BFCL",
        category: Optional[str] = None
    ):
        """Initialize the BFCL dataset loader.

        Args:
            bfcl_data_dir: BFCL official data directory path (containing BFCL_v4_*.json files).
            category: Evaluation category, e.g. 'simple_python', 'multiple', etc.
        """
        self.bfcl_data_dir = Path(bfcl_data_dir)
        self.category = category
        self.data = []
        self.ground_truth = {}

        # Validate data directory
        if not self.bfcl_data_dir.exists():
            print(f"   ⚠️ BFCL data directory does not exist: {self.bfcl_data_dir}")
            print(f"   Please ensure the BFCL repository has been cloned to the correct location")

        # Validate possible_answer directory
        self.answer_dir = self.bfcl_data_dir / "possible_answer"
        if not self.answer_dir.exists():
            print(f"   ⚠️ Ground truth directory does not exist: {self.answer_dir}")

    def load(self) -> List[Dict[str, Any]]:
        """Load the dataset (including test data and ground truth).

        Returns:
            Dataset list, where each element contains the question, function definitions, ground truth, etc.
        """
        if not self.bfcl_data_dir.exists():
            print(f"   ⚠️ Data directory does not exist, unable to load data")
            return []

        # Determine which files to load
        if self.category:
            # Load specified category
            filename = self.CATEGORY_MAPPING.get(self.category)
            if not filename:
                print(f"   ⚠️ Unknown category: {self.category}")
                print(f"   Supported categories: {list(self.CATEGORY_MAPPING.keys())}")
                return []

            self.data = self._load_category(filename)
        else:
            # Load all categories (not recommended due to large data volume)
            print(f"   ⚠️ No category specified, loading simple_python as an example")
            self.data = self._load_category(self.CATEGORY_MAPPING["simple_python"])

        print(f"✅ BFCL dataset loaded successfully")
        print(f"   Data directory: {self.bfcl_data_dir}")
        print(f"   Category: {self.category or 'simple_python'}")
        print(f"   Sample count: {len(self.data)}")
        print(f"   Ground truth count: {len(self.ground_truth)}")

        return self.data

    def _load_category(self, filename: str) -> List[Dict[str, Any]]:
        """Load data for a specified category (including test data and ground truth).

        Args:
            filename: File name (without .json extension), e.g. 'BFCL_v4_simple_python'.

        Returns:
            List of test data.
        """
        # Load test data
        test_file = self.bfcl_data_dir / f"{filename}.json"
        if not test_file.exists():
            print(f"   ⚠️ Test data file does not exist: {test_file}")
            return []

        test_data = self._load_jsonl_file(test_file)
        print(f"   ✓ Loaded test data: {test_file.name} ({len(test_data)} samples)")

        # Load ground truth
        gt_file = self.answer_dir / f"{filename}.json"
        if gt_file.exists():
            gt_data = self._load_jsonl_file(gt_file)
            # Build ground truth dictionary
            for item in gt_data:
                item_id = item.get("id")
                if item_id:
                    self.ground_truth[item_id] = item.get("ground_truth", [])
            print(f"   ✓ Loaded ground truth: {gt_file.name} ({len(gt_data)} samples)")
        else:
            print(f"   ⚠️ Ground truth file does not exist: {gt_file}")

        # Merge test data with ground truth
        for item in test_data:
            item_id = item.get("id")
            if item_id and item_id in self.ground_truth:
                item["ground_truth"] = self.ground_truth[item_id]

        return test_data

    def _load_jsonl_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load a JSONL file (one JSON object per line).

        Args:
            file_path: JSON/JSONL file path.

        Returns:
            List of data.
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"   ⚠️ JSON parsing failed: {e}")
                        continue
        return data

    def get_ground_truth(self, sample_id: str) -> List[Dict[str, Any]]:
        """Get the ground truth for a specified sample.

        Args:
            sample_id: Sample ID.

        Returns:
            Ground truth list.
        """
        return self.ground_truth.get(sample_id, [])

    def get_sample(self, index: int) -> Dict[str, Any]:
        """Get a single sample.

        Args:
            index: Sample index.

        Returns:
            Sample data.
        """
        if not self.data:
            self.load()
        return self.data[index] if index < len(self.data) else {}

    def get_available_categories(self) -> List[str]:
        """Get all available categories.

        Returns:
            List of categories.
        """
        return list(self.CATEGORY_MAPPING.keys())

    def __len__(self) -> int:
        """Return the dataset size."""
        if not self.data:
            self.load()
        return len(self.data)

    def __iter__(self):
        """Iterator."""
        if not self.data:
            self.load()
        return iter(self.data)
