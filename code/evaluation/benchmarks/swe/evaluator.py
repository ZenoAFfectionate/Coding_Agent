"""
SWE-bench 评估器模块

负责评估智能体在 SWE-bench 基准测试上的表现。

每个实例代表一个真实的 GitHub Issue，评估器会:
1. 克隆对应仓库并切换到 base_commit
2. 构建 prompt（issue + hints）
3. 运行智能体
4. 通过 git diff 收集智能体产生的补丁
5. 将预测补丁与 gold patch 进行比较（可选：运行测试）
"""

from typing import Dict, Any, List, Optional, Union, Callable
import time
import json
import subprocess
import tempfile
import shutil
from pathlib import Path

from code.evaluation.benchmarks.swe.dataset import SWEDataset
from code.evaluation.benchmarks.swe.metrics import SWEMetrics


class SWEEvaluator:
    """SWE-bench 评估器

    评估智能体修复真实 GitHub Issue 的能力。

    Attributes:
        dataset: SWE-bench 数据集
        metrics: 指标计算器
        workspace_base: 临时工作目录的父目录
        timeout_per_instance: 每个实例的超时时间(秒)
        run_tests: 是否运行 FAIL_TO_PASS 测试
    """

    def __init__(
        self,
        dataset: Optional[SWEDataset] = None,
        workspace_base: Optional[str] = None,
        timeout_per_instance: int = 600,
        run_tests: bool = False,
    ):
        """初始化 SWE-bench 评估器

        Args:
            dataset: SWE-bench 数据集，为 None 则自动创建 (dev split)
            workspace_base: 克隆仓库的基础目录，默认使用系统临时目录
            timeout_per_instance: 每个实例的超时时间(秒)
            run_tests: 是否执行 FAIL_TO_PASS 测试验证
        """
        self.dataset = dataset if dataset is not None else SWEDataset()
        self.metrics = SWEMetrics()
        self.workspace_base = workspace_base
        self.timeout_per_instance = timeout_per_instance
        self.run_tests = run_tests

    def evaluate(
        self,
        agent_factory: Callable[..., Any],
        max_samples: Optional[int] = None,
        **agent_kwargs,
    ) -> Dict[str, Any]:
        """评估智能体

        因为每个 SWE-bench 实例需要不同的 workspace（克隆的仓库），
        所以传入的是 agent_factory (如 build_agent)，评估器会为每个
        实例创建新的 agent。

        Args:
            agent_factory: 接受 workspace 关键字参数并返回 agent 的工厂函数
            max_samples: 最大评估样本数，None 表示评估全部
            **agent_kwargs: 传递给 agent_factory 的额外参数

        Returns:
            评估结果字典
        """
        print(f"\n🔧 开始 SWE-bench 评估...")

        # 加载数据集
        dataset = self.dataset.load()
        if not dataset:
            print("   ⚠️ 数据集为空，跳过评估")
            return self._create_empty_results()

        # 限制样本数量
        if max_samples:
            dataset = dataset[:max_samples]

        print(f"   样本数量: {len(dataset)}")
        print(f"   运行测试: {'是' if self.run_tests else '否'}")

        results: List[Dict[str, Any]] = []
        for i, sample in enumerate(dataset):
            print(
                f"   进度: {i + 1}/{len(dataset)} - {sample.get('instance_id', '')}"
            )

            try:
                sample_result = self.evaluate_sample(
                    agent_factory, sample, **agent_kwargs
                )
                results.append(sample_result)
            except Exception as e:
                print(f"   ⚠️ 实例 {sample.get('instance_id')} 评估失败: {e}")
                results.append(
                    {
                        "instance_id": sample.get("instance_id", ""),
                        "repo": sample.get("repo", ""),
                        "exact_match": False,
                        "tests_passed": False,
                        "patch_metrics": {},
                        "predicted_patch": "",
                        "error": str(e),
                        "score": 0.0,
                    }
                )

        # 计算综合指标
        overall_metrics = self.metrics.compute_metrics(results)

        final_results = {
            "benchmark": "SWE-bench",
            "total_samples": len(results),
            "resolved_rate": overall_metrics["resolved_rate"],
            "exact_match_rate": overall_metrics["exact_match_rate"],
            "average_line_overlap": overall_metrics["average_line_overlap"],
            "average_execution_time": overall_metrics["average_execution_time"],
            "repo_metrics": overall_metrics["repo_metrics"],
            "detailed_results": results,
        }

        print(f"✅ SWE-bench 评估完成")
        print(f"   解决率: {overall_metrics['resolved_rate']:.2%}")
        print(f"   精确匹配率: {overall_metrics['exact_match_rate']:.2%}")
        print(f"   平均行重叠度: {overall_metrics['average_line_overlap']:.2%}")

        return final_results

    def evaluate_sample(
        self,
        agent_factory: Callable[..., Any],
        sample: Dict[str, Any],
        **agent_kwargs,
    ) -> Dict[str, Any]:
        """评估单个实例

        Args:
            agent_factory: agent 工厂函数
            sample: SWE-bench 实例
            **agent_kwargs: 传递给 agent_factory 的额外参数

        Returns:
            单个实例的评估结果
        """
        instance_id = sample.get("instance_id", "")
        workspace = None

        try:
            # 1. 克隆仓库并切换到 base_commit
            workspace = self._setup_repo(sample)

            # 2. 创建 agent（workspace 指向克隆出的仓库）
            agent = agent_factory(workspace=str(workspace), **agent_kwargs)

            # 3. 构建 prompt
            prompt = self._build_prompt(sample)

            # 4. 运行 agent
            start_time = time.time()
            agent.run(prompt)
            execution_time = time.time() - start_time

            # 5. 收集 agent 产生的 patch
            predicted_patch = self._collect_patch(workspace)

            # 6. 计算补丁指标
            gold_patch = sample.get("patch", "")
            patch_metrics = self.metrics.calculate_patch_metrics(
                predicted_patch, gold_patch
            )

            # 7. 可选：运行测试
            tests_passed = False
            test_output = ""
            if self.run_tests and sample.get("FAIL_TO_PASS"):
                tests_passed, test_output = self._run_tests(
                    workspace, sample["FAIL_TO_PASS"]
                )

            score = 1.0 if patch_metrics["exact_match"] or tests_passed else 0.0

            return {
                "instance_id": instance_id,
                "repo": sample.get("repo", ""),
                "exact_match": patch_metrics["exact_match"],
                "tests_passed": tests_passed,
                "patch_metrics": patch_metrics,
                "predicted_patch": predicted_patch,
                "score": score,
                "execution_time": execution_time,
                "test_output": test_output,
            }

        except Exception as e:
            return {
                "instance_id": instance_id,
                "repo": sample.get("repo", ""),
                "exact_match": False,
                "tests_passed": False,
                "patch_metrics": {},
                "predicted_patch": "",
                "score": 0.0,
                "error": str(e),
            }
        finally:
            if workspace:
                self._cleanup_workspace(workspace)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup_repo(self, sample: Dict[str, Any]) -> Path:
        """克隆仓库并切换到 base_commit

        Args:
            sample: SWE-bench 实例

        Returns:
            克隆出的仓库路径
        """
        repo = sample["repo"]
        base_commit = sample["base_commit"]

        workspace = Path(
            tempfile.mkdtemp(
                prefix=f"swe_{sample.get('instance_id', 'unknown')}_",
                dir=self.workspace_base,
            )
        )

        repo_url = f"https://github.com/{repo}.git"

        # 浅克隆 + checkout 目标 commit
        subprocess.run(
            ["git", "clone", "--no-checkout", repo_url, str(workspace)],
            check=True,
            capture_output=True,
            timeout=300,
        )
        subprocess.run(
            ["git", "checkout", base_commit],
            check=True,
            capture_output=True,
            cwd=str(workspace),
            timeout=60,
        )

        return workspace

    def _collect_patch(self, workspace: Path) -> str:
        """收集 agent 在工作目录中的所有更改

        Args:
            workspace: 仓库工作目录

        Returns:
            unified diff 字符串
        """
        result = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(workspace),
            timeout=30,
        )
        return result.stdout

    def _build_prompt(self, sample: Dict[str, Any]) -> str:
        """构建发送给 agent 的 prompt

        Args:
            sample: SWE-bench 实例

        Returns:
            prompt 字符串
        """
        problem = sample.get("problem_statement", "")
        hints = sample.get("hints_text", "")
        repo = sample.get("repo", "")
        instance_id = sample.get("instance_id", "")

        prompt = (
            f"You are working on the repository: {repo}\n"
            f"Instance ID: {instance_id}\n\n"
            f"## GitHub Issue\n\n{problem}\n"
        )

        if hints:
            prompt += f"\n## Hints\n\n{hints}\n"

        prompt += (
            "\n## Instructions\n\n"
            "Please investigate this issue in the codebase and produce a fix. "
            "Explore the relevant source files, understand the root cause, and "
            "make the necessary code changes to resolve the issue. "
            "Do NOT run tests or create new test files — only modify source code."
        )

        return prompt

    def _run_tests(
        self, workspace: Path, fail_to_pass: List[str]
    ) -> tuple:
        """运行 FAIL_TO_PASS 测试列表

        Args:
            workspace: 仓库工作目录
            fail_to_pass: 需要从 FAIL 变为 PASS 的测试列表

        Returns:
            (all_passed, test_output)
        """
        if not fail_to_pass:
            return False, ""

        try:
            result = subprocess.run(
                ["python", "-m", "pytest"] + fail_to_pass + ["-x", "--tb=short"],
                capture_output=True,
                text=True,
                cwd=str(workspace),
                timeout=self.timeout_per_instance,
            )
            all_passed = result.returncode == 0
            return all_passed, result.stdout + result.stderr
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return False, str(e)

    def _cleanup_workspace(self, workspace: Path) -> None:
        """清理临时工作目录"""
        try:
            shutil.rmtree(str(workspace), ignore_errors=True)
        except Exception:
            pass

    def _create_empty_results(self) -> Dict[str, Any]:
        """创建空的评估结果"""
        return {
            "benchmark": "SWE-bench",
            "total_samples": 0,
            "resolved_rate": 0.0,
            "exact_match_rate": 0.0,
            "average_line_overlap": 0.0,
            "average_execution_time": 0.0,
            "repo_metrics": {},
            "detailed_results": [],
        }

    def export_to_swe_format(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path],
    ) -> None:
        """导出为 SWE-bench 官方提交格式

        JSONL 格式，每行包含 instance_id 和 model_patch。

        Args:
            results: evaluate() 返回的结果字典
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        detailed = results.get("detailed_results", [])

        with open(output_path, "w", encoding="utf-8") as f:
            for r in detailed:
                entry = {
                    "instance_id": r.get("instance_id", ""),
                    "model_patch": r.get("predicted_patch", ""),
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"✅ SWE-bench 格式结果已导出")
        print(f"   输出文件: {output_path}")
        print(f"   样本数: {len(detailed)}")
