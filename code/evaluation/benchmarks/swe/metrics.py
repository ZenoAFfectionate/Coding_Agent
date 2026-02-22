"""
SWE-bench 评估指标模块

计算 SWE-bench 相关的评估指标，包括补丁匹配和仓库维度汇总。
"""

from typing import Dict, Any, List, Set
import re


class SWEMetrics:
    """SWE-bench 评估指标计算器

    计算软件工程修复相关的评估指标:
    - 解决率 (Resolved Rate): 补丁匹配或测试通过的比例
    - 补丁指标: 文件级别和行级别的重叠度
    - 仓库维度指标: 按仓库分组的解决率
    """

    @staticmethod
    def calculate_resolved_rate(results: List[Dict[str, Any]]) -> float:
        """计算解决率

        如果实例的 exact_match 为 True 或 tests_passed 为 True，则视为已解决。

        Args:
            results: 每个实例的评估结果列表

        Returns:
            解决率 (0-1)
        """
        if not results:
            return 0.0

        resolved = sum(
            1
            for r in results
            if r.get("exact_match", False) or r.get("tests_passed", False)
        )
        return resolved / len(results)

    @staticmethod
    def calculate_patch_metrics(
        predicted_patch: str, gold_patch: str
    ) -> Dict[str, Any]:
        """计算两个补丁之间的匹配指标

        Args:
            predicted_patch: 智能体生成的补丁
            gold_patch: 标准答案补丁

        Returns:
            包含 exact_match / files_matched / line_overlap 的指标字典
        """
        exact_match = predicted_patch.strip() == gold_patch.strip()

        pred_files = SWEMetrics._extract_changed_files(predicted_patch)
        gold_files = SWEMetrics._extract_changed_files(gold_patch)

        if gold_files:
            file_intersection = pred_files & gold_files
            file_precision = len(file_intersection) / len(pred_files) if pred_files else 0.0
            file_recall = len(file_intersection) / len(gold_files)
        else:
            file_precision = 0.0
            file_recall = 0.0

        files_matched = pred_files == gold_files

        pred_lines = SWEMetrics._extract_changed_lines(predicted_patch)
        gold_lines = SWEMetrics._extract_changed_lines(gold_patch)

        if pred_lines or gold_lines:
            intersection = pred_lines & gold_lines
            union = pred_lines | gold_lines
            line_overlap = len(intersection) / len(union) if union else 0.0
        else:
            line_overlap = 1.0 if exact_match else 0.0

        return {
            "exact_match": exact_match,
            "files_matched": files_matched,
            "file_precision": file_precision,
            "file_recall": file_recall,
            "line_overlap": line_overlap,
            "predicted_files": sorted(pred_files),
            "gold_files": sorted(gold_files),
        }

    @staticmethod
    def calculate_repo_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """按仓库分组计算指标

        Args:
            results: 每个实例的评估结果列表

        Returns:
            以仓库名为键的指标字典
        """
        repo_groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in results:
            repo = r.get("repo", "unknown")
            repo_groups.setdefault(repo, []).append(r)

        repo_metrics: Dict[str, Any] = {}
        for repo, group in sorted(repo_groups.items()):
            resolved = sum(
                1
                for r in group
                if r.get("exact_match", False) or r.get("tests_passed", False)
            )
            repo_metrics[repo] = {
                "total": len(group),
                "resolved": resolved,
                "resolved_rate": resolved / len(group) if group else 0.0,
            }

        return repo_metrics

    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算综合指标

        Args:
            results: 评估结果列表

        Returns:
            完整的指标字典
        """
        if not results:
            return self._empty_metrics()

        resolved_rate = self.calculate_resolved_rate(results)
        repo_metrics = self.calculate_repo_metrics(results)

        # 汇总补丁指标
        exact_matches = sum(1 for r in results if r.get("exact_match", False))
        files_matched = sum(
            1 for r in results if r.get("patch_metrics", {}).get("files_matched", False)
        )
        line_overlaps = [
            r.get("patch_metrics", {}).get("line_overlap", 0.0) for r in results
        ]
        avg_line_overlap = (
            sum(line_overlaps) / len(line_overlaps) if line_overlaps else 0.0
        )

        # 执行时间统计
        exec_times = [
            r.get("execution_time", 0.0) for r in results if "execution_time" in r
        ]
        avg_execution_time = sum(exec_times) / len(exec_times) if exec_times else 0.0

        return {
            "total_samples": len(results),
            "resolved_rate": resolved_rate,
            "exact_matches": exact_matches,
            "exact_match_rate": exact_matches / len(results),
            "files_matched_count": files_matched,
            "average_line_overlap": avg_line_overlap,
            "average_execution_time": avg_execution_time,
            "repo_metrics": repo_metrics,
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        """返回空指标"""
        return {
            "total_samples": 0,
            "resolved_rate": 0.0,
            "exact_matches": 0,
            "exact_match_rate": 0.0,
            "files_matched_count": 0,
            "average_line_overlap": 0.0,
            "average_execution_time": 0.0,
            "repo_metrics": {},
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_changed_files(patch: str) -> Set[str]:
        """从 unified-diff 补丁中提取被修改的文件路径"""
        files: Set[str] = set()
        for m in re.finditer(r"^diff --git a/(.+?) b/(.+?)$", patch, re.MULTILINE):
            files.add(m.group(2))
        return files

    @staticmethod
    def _extract_changed_lines(patch: str) -> Set[str]:
        """从补丁中提取所有新增/删除行 (去掉前缀空格后)

        用于计算 Jaccard 相似度。
        """
        lines: Set[str] = set()
        for line in patch.splitlines():
            stripped = line.rstrip()
            if stripped.startswith("+") and not stripped.startswith("+++"):
                lines.add(stripped)
            elif stripped.startswith("-") and not stripped.startswith("---"):
                lines.add(stripped)
        return lines
