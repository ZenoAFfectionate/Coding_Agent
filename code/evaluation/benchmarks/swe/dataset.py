"""
SWE-bench 数据集加载模块

负责加载 SWE-bench (Software Engineering Benchmark) 数据集,
用于评估智能体修复真实 GitHub Issue 的能力。
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json


class SWEDataset:
    """SWE-bench 数据集加载器

    从本地 JSONL 文件加载 SWE-bench 数据集。

    SWE-bench 是一个软件工程评估基准，每个实例包含一个 GitHub Issue
    (problem_statement)、对应的代码仓库状态 (repo + base_commit)、
    以及用于验证的测试列表 (FAIL_TO_PASS / PASS_TO_PASS)。

    Attributes:
        split: 数据集分割 (dev/test/train)
        data_dir: 数据目录路径
        repo_filter: 可选的仓库名过滤器
        data: 加载的数据列表
    """

    def __init__(
        self,
        split: str = "dev",
        data_dir: Optional[Union[str, Path]] = None,
        repo_filter: Optional[str] = None,
    ):
        """初始化 SWE-bench 数据集加载器

        Args:
            split: 数据集分割 (dev/test/train)
            data_dir: JSONL 数据文件所在目录，默认 data/SWE
            repo_filter: 仅保留指定仓库的实例 (e.g. "astropy/astropy")
        """
        self.split = split
        self.data_dir = Path(data_dir) if data_dir else Path("data/SWE")
        self.repo_filter = repo_filter
        self.data: List[Dict[str, Any]] = []

    def load(self) -> List[Dict[str, Any]]:
        """加载数据集

        Returns:
            数据集列表，每个元素为标准化后的 SWE-bench 实例
        """
        jsonl_path = self.data_dir / f"{self.split}.jsonl"

        if not jsonl_path.exists():
            print(f"   ⚠️ 数据文件不存在: {jsonl_path}")
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

        # 按仓库过滤
        if self.repo_filter:
            self.data = [
                item for item in self.data if item.get("repo") == self.repo_filter
            ]

        print(f"✅ SWE-bench 数据集加载完成")
        print(f"   分割: {self.split}")
        print(f"   仓库过滤: {self.repo_filter or '全部'}")
        print(f"   样本数: {len(self.data)}")

        return self.data

    def _standardize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """标准化数据项格式

        将原始 JSONL 字段映射为统一 schema。
        """
        fail_to_pass = item.get("FAIL_TO_PASS", "[]")
        pass_to_pass = item.get("PASS_TO_PASS", "[]")

        # FAIL_TO_PASS / PASS_TO_PASS 在 JSONL 中可能是 JSON 字符串
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
        """获取单个样本

        Args:
            index: 样本索引

        Returns:
            样本数据
        """
        if not self.data:
            self.load()
        return self.data[index] if index < len(self.data) else {}

    def get_by_repo(self, repo: str) -> List[Dict[str, Any]]:
        """获取指定仓库的所有实例

        Args:
            repo: 仓库全名 (e.g. "django/django")

        Returns:
            该仓库的所有样本
        """
        if not self.data:
            self.load()
        return [item for item in self.data if item.get("repo") == repo]

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息

        Returns:
            统计信息字典
        """
        if not self.data:
            self.load()

        # 按仓库统计
        repo_counts: Dict[str, int] = {}
        for item in self.data:
            repo = item.get("repo", "unknown")
            repo_counts[repo] = repo_counts.get(repo, 0) + 1

        # 按版本统计
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
        """返回数据集大小"""
        if not self.data:
            self.load()
        return len(self.data)

    def __bool__(self) -> bool:
        """Dataset object is always truthy (prevents falsy on empty data)."""
        return True

    def __iter__(self):
        """迭代器"""
        if not self.data:
            self.load()
        return iter(self.data)
