"""
GAIA Dataset Loading Module

Loads the GAIA (General AI Assistants) dataset from HuggingFace or local files.
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json


class GAIADataset:
    """GAIA Dataset Loader

    Loads the GAIA dataset from HuggingFace, supporting different difficulty levels.

    GAIA is a general AI assistant evaluation benchmark containing 466 real-world
    questions that require reasoning, multimodal processing, web browsing, and
    tool-use capabilities.

    Difficulty levels:
    - Level 1: Simple questions (0-step reasoning, direct answer)
    - Level 2: Medium questions (1-5 step reasoning, simple tool use)
    - Level 3: Complex questions (5+ step reasoning, complex tool chains)

    Attributes:
        dataset_name: HuggingFace dataset name
        split: Dataset split (validation/test)
        level: Difficulty level
        data: Loaded data list
    """

    def __init__(
        self,
        dataset_name: str = "gaia-benchmark/GAIA",
        split: str = "validation",
        level: Optional[int] = None,
        local_data_dir: Optional[Union[str, Path]] = None
    ):
        """Initialize the GAIA dataset loader.

        Args:
            dataset_name: HuggingFace dataset name
            split: Dataset split (validation/test)
            level: Difficulty level (1-3), None to load all levels
            local_data_dir: Local data directory path
        """
        self.dataset_name = dataset_name
        self.split = split
        self.level = level
        self.local_data_dir = Path(local_data_dir) if local_data_dir else None
        self.data = []
        self._is_local = self._check_if_local_source()

    def _check_if_local_source(self) -> bool:
        """Check whether to use a local data source."""
        if self.local_data_dir and self.local_data_dir.exists():
            return True
        return False

    def load(self) -> List[Dict[str, Any]]:
        """Load the dataset.

        Returns:
            List of dataset items, each containing question, answer, difficulty, etc.
        """
        if self._is_local:
            self.data = self._load_from_local()
        else:
            self.data = self._load_from_huggingface()

        # Filter by level
        if self.level is not None:
            self.data = [item for item in self.data if item.get("level") == self.level]

        print(f"✅ GAIA dataset loaded")
        print(f"   Source: {self.dataset_name}")
        print(f"   Split: {self.split}")
        print(f"   Level: {self.level or 'all'}")
        print(f"   Samples: {len(self.data)}")

        return self.data

    def _load_from_local(self) -> List[Dict[str, Any]]:
        """Load dataset from local files (supports parquet and JSON formats)."""
        data = []

        if not self.local_data_dir or not self.local_data_dir.exists():
            print("   ⚠️ Local data directory does not exist")
            return data

        # Prefer parquet format (default for HuggingFace snapshot_download)
        split_dir = self.local_data_dir / self.split
        if split_dir.exists():
            parquet_file = split_dir / "metadata.parquet"
            if parquet_file.exists():
                return self._load_parquet(parquet_file, split_dir)

        # Fallback: look for parquet in root directory
        parquet_file = self.local_data_dir / "metadata.parquet"
        if parquet_file.exists():
            return self._load_parquet(parquet_file, self.local_data_dir)

        # Fallback: look for JSONL files (legacy GAIA 2023 format)
        for subdir in [self.local_data_dir / "2023" / self.split, split_dir, self.local_data_dir]:
            jsonl_file = subdir / "metadata.jsonl"
            if jsonl_file.exists():
                return self._load_jsonl(jsonl_file, subdir)

        # Final fallback: look for JSON files
        json_files = list(self.local_data_dir.glob("**/*.json"))
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                if isinstance(file_data, list):
                    for item in file_data:
                        data.append(self._standardize_item(item))
                else:
                    data.append(self._standardize_item(file_data))
                print(f"   Loaded file: {json_file.name} ({len(file_data) if isinstance(file_data, list) else 1} samples)")
            except Exception as e:
                print(f"   ⚠️ Failed to load file: {json_file.name} - {e}")

        return data

    def _load_parquet(self, parquet_file: Path, attachments_dir: Path) -> List[Dict[str, Any]]:
        """Load data from a parquet file."""
        import pandas as pd

        df = pd.read_parquet(parquet_file)
        print(f"   Loaded file: {parquet_file} ({len(df)} samples)")

        data = []
        for _, row in df.iterrows():
            item = row.to_dict()

            # Skip placeholder entries
            if item.get("task_id") == "0-0-0-0-0":
                continue

            # Resolve attachment file paths to absolute paths
            if item.get("file_name"):
                abs_path = attachments_dir / item["file_name"]
                if abs_path.exists():
                    item["file_name"] = str(abs_path)

            data.append(self._standardize_item(item))

        return data

    def _load_jsonl(self, jsonl_file: Path, attachments_dir: Path) -> List[Dict[str, Any]]:
        """Load data from a JSONL file."""
        data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if item.get("task_id") == "0-0-0-0-0":
                    continue
                if item.get("file_name"):
                    abs_path = attachments_dir / item["file_name"]
                    if abs_path.exists():
                        item["file_name"] = str(abs_path)
                data.append(self._standardize_item(item))

        print(f"   Loaded file: {jsonl_file} ({len(data)} samples)")
        return data

    def _load_from_huggingface(self) -> List[Dict[str, Any]]:
        """Download and load the GAIA dataset from HuggingFace.

        Note: GAIA is a gated dataset and requires the HF_TOKEN environment variable.
        Uses snapshot_download to download the entire dataset locally.
        """
        try:
            from huggingface_hub import snapshot_download
            import os
            import json
            from pathlib import Path

            print(f"   Downloading from HuggingFace: {self.dataset_name}")

            # Get HF token
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                print("   ⚠️ HF_TOKEN environment variable not found")
                print("   GAIA is a gated dataset; request access on HuggingFace first")
                print("   Then set the environment variable: HF_TOKEN=your_token")
                return []

            # Download dataset locally
            print(f"   📥 Downloading GAIA dataset...")
            local_dir = Path.cwd() / "data" / "gaia"
            local_dir.mkdir(parents=True, exist_ok=True)

            try:
                snapshot_download(
                    repo_id=self.dataset_name,
                    repo_type="dataset",
                    local_dir=str(local_dir),
                    token=hf_token,
                    local_dir_use_symlinks=False
                )
                print(f"   ✓ Dataset download complete: {local_dir}")
            except Exception as e:
                print(f"   ⚠️ Download failed: {e}")
                print("   Please ensure:")
                print("   1. You have requested access to GAIA on HuggingFace")
                print("   2. Your HF_TOKEN is correct and valid")
                return []

            # Read metadata.jsonl file
            metadata_file = local_dir / "2023" / self.split / "metadata.jsonl"
            if not metadata_file.exists():
                print(f"   ⚠️ Metadata file not found: {metadata_file}")
                return []

            # Load data
            data = []
            with open(metadata_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    item = json.loads(line)

                    # Skip placeholder entries
                    if item.get("task_id") == "0-0-0-0-0":
                        continue

                    # Resolve file paths
                    if item.get("file_name"):
                        item["file_name"] = str(local_dir / "2023" / self.split / item["file_name"])

                    # Standardize and append
                    standardized_item = self._standardize_item(item)
                    data.append(standardized_item)

            print(f"   ✓ Loaded {len(data)} samples")
            return data

        except ImportError:
            print("   ⚠️ huggingface_hub is not installed")
            print("   Hint: pip install huggingface_hub")
            return []
        except Exception as e:
            print(f"   ⚠️ Loading failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _standardize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize a data item to a common format."""
        raw_level = item.get("Level", item.get("level", 1))
        try:
            level_int = int(raw_level)
        except (ValueError, TypeError):
            level_int = 1

        standardized = {
            "task_id": item.get("task_id", ""),
            "question": item.get("Question", item.get("question", "")),
            "level": level_int,
            "final_answer": item.get("Final answer", item.get("final_answer", "")),
            "file_name": item.get("file_name", ""),
            "file_path": item.get("file_path", ""),
            "annotator_metadata": item.get("Annotator Metadata", item.get("annotator_metadata", {})),
            "steps": item.get("Steps", item.get("steps", 0)),
            "tools": item.get("Tools", item.get("tools", [])),
            "raw_item": item  # Keep original data
        }

        return standardized

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

    def get_by_level(self, level: int) -> List[Dict[str, Any]]:
        """Get samples of a specific difficulty level.

        Args:
            level: Difficulty level (1-3)

        Returns:
            All samples at that level
        """
        if not self.data:
            self.load()
        return [item for item in self.data if item.get("level") == level]

    def get_level_distribution(self) -> Dict[int, int]:
        """Get the distribution of difficulty levels.

        Returns:
            Dict mapping level to sample count
        """
        if not self.data:
            self.load()

        distribution = {1: 0, 2: 0, 3: 0}
        for item in self.data:
            level = item.get("level", 1)
            if level in distribution:
                distribution[level] += 1

        return distribution

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics.

        Returns:
            Statistics dict
        """
        if not self.data:
            self.load()

        level_dist = self.get_level_distribution()

        # Count samples with file attachments
        with_files = sum(1 for item in self.data if item.get("file_name"))

        # Calculate average steps
        steps_list = [item.get("steps", 0) for item in self.data if item.get("steps")]
        avg_steps = sum(steps_list) / len(steps_list) if steps_list else 0

        return {
            "total_samples": len(self.data),
            "level_distribution": level_dist,
            "samples_with_files": with_files,
            "average_steps": avg_steps,
            "split": self.split
        }

    def __len__(self) -> int:
        """Return dataset size."""
        if not self.data:
            self.load()
        return len(self.data)

    def __iter__(self):
        """Iterator."""
        if not self.data:
            self.load()
        return iter(self.data)
