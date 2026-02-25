"""
GAIA 评估器模块

负责评估智能体在 GAIA 基准测试上的表现
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import time
import re
import json
import logging
import base64
from pathlib import Path
from code.evaluation.benchmarks.gaia.dataset import GAIADataset
from code.evaluation.benchmarks.gaia.metrics import GAIAMetrics

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


class GAIAEvaluator:
    """GAIA 评估器

    评估智能体的通用AI助手能力,包括:
    - 问题理解和推理
    - 多步骤问题解决
    - 工具使用能力
    - 答案准确性

    GAIA评估采用严格的答案匹配标准:
    - 精确匹配: 答案完全一致
    - 部分匹配: 答案包含正确信息但格式不同

    Attributes:
        dataset: GAIA 数据集
        metrics: 评估指标计算器
        level: 难度级别
        strict_mode: 是否使用严格匹配模式
    """

    def __init__(
        self,
        dataset: Optional[GAIADataset] = None,
        level: Optional[int] = None,
        local_data_dir: Optional[str] = None,
        strict_mode: bool = True,
        llm=None,
    ):
        """初始化 GAIA 评估器

        Args:
            dataset: GAIA 数据集,如果为 None 则自动创建
            level: 难度级别 (1-3)
            local_data_dir: 本地数据目录
            strict_mode: 是否使用严格匹配模式
            llm: Optional HelloAgentsLLM instance for direct LLM mode
                 (bypasses the ReAct agent loop).
        """
        self.dataset = dataset or GAIADataset(
            level=level,
            local_data_dir=local_data_dir
        )
        self.metrics = GAIAMetrics()
        self.level = level
        self.strict_mode = strict_mode
        self.llm = llm

        # Load benchmark-specific prompts
        self.gaia_system_prompt = _load_prompt(EVAL_PROMPTS_DIR / "gaia_system.prompt")
        self.task_template = _load_prompt(EVAL_PROMPTS_DIR / "gaia_task.prompt")
        
    def evaluate(self, agent: Any = None, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """评估智能体

        Args:
            agent: 要评估的智能体 (used only when self.llm is None)
            max_samples: 最大评估样本数,None表示评估全部

        Returns:
            评估结果字典,包含各项指标
        """
        mode_label = "direct LLM" if self.llm else "agent"
        agent_name = getattr(self.llm, 'model', None) or getattr(agent, 'name', 'Unknown')

        print(f"\n🌟 开始 GAIA 评估...")
        print(f"   模式: {mode_label}")
        print(f"   模型/智能体: {agent_name}")
        print(f"   难度级别: {self.level or '全部'}")
        print(f"   匹配模式: {'严格' if self.strict_mode else '宽松'}")

        # 加载数据集
        dataset = self.dataset.load()
        if not dataset:
            print("   ⚠️ 数据集为空,跳过评估")
            return self._create_empty_results(agent)

        # 限制样本数量
        if max_samples:
            dataset = dataset[:max_samples]

        print(f"   样本数量: {len(dataset)}")

        # 执行评估
        results = []
        skipped_samples = 0
        level_stats = {1: {"total": 0, "correct": 0, "partial": 0},
                      2: {"total": 0, "correct": 0, "partial": 0},
                      3: {"total": 0, "correct": 0, "partial": 0}}

        for i, sample in enumerate(dataset):
            if i % 10 == 0:
                print(f"   进度: {i+1}/{len(dataset)}")

            try:
                sample_result = self.evaluate_sample(agent, sample)
                results.append(sample_result)

                # Track skipped samples
                if sample_result.get("skipped"):
                    skipped_samples += 1
                    continue

                # 按级别统计
                level = sample.get("level", 1)
                if level in level_stats:
                    level_stats[level]["total"] += 1
                    if sample_result["exact_match"]:
                        level_stats[level]["correct"] += 1
                    if sample_result["partial_match"]:
                        level_stats[level]["partial"] += 1

            except Exception as e:
                print(f"   ⚠️ 样本 {i} 评估失败: {e}")
                results.append({
                    "exact_match": False,
                    "partial_match": False,
                    "predicted": None,
                    "expected": sample.get("final_answer"),
                    "error": str(e),
                    "score": 0.0
                })

        # 计算总体指标 (skipped samples excluded from totals)
        evaluated_results = [r for r in results if not r.get("skipped")]
        total_samples = len(evaluated_results)
        exact_matches = sum(1 for r in evaluated_results if r["exact_match"])
        partial_matches = sum(1 for r in evaluated_results if r["partial_match"])

        exact_match_rate = exact_matches / total_samples if total_samples > 0 else 0.0
        partial_match_rate = partial_matches / total_samples if total_samples > 0 else 0.0

        # 计算分级指标
        level_metrics = {}
        for level, stats in level_stats.items():
            if stats["total"] > 0:
                level_metrics[f"Level_{level}"] = {
                    "total": stats["total"],
                    "exact_matches": stats["correct"],
                    "partial_matches": stats["partial"],
                    "exact_match_rate": stats["correct"] / stats["total"],
                    "partial_match_rate": stats["partial"] / stats["total"]
                }

        final_results = {
            "benchmark": "GAIA",
            "agent_name": agent_name,
            "strict_mode": self.strict_mode,
            "level_filter": self.level,
            "total_samples": total_samples,
            "skipped_samples": skipped_samples,
            "exact_matches": exact_matches,
            "partial_matches": partial_matches,
            "exact_match_rate": exact_match_rate,
            "partial_match_rate": partial_match_rate,
            "level_metrics": level_metrics,
            "detailed_results": results
        }

        print(f"✅ GAIA 评估完成")
        print(f"   评估样本: {total_samples} (跳过: {skipped_samples})")
        print(f"   精确匹配率: {exact_match_rate:.2%}")
        print(f"   部分匹配率: {partial_match_rate:.2%}")
        for level_name, metrics in level_metrics.items():
            print(f"   {level_name}: {metrics['exact_match_rate']:.2%} 精确 / {metrics['partial_match_rate']:.2%} 部分")

        return final_results
    
    def evaluate_sample(self, agent: Any, sample: Dict[str, Any]) -> Dict[str, Any]:
        """评估单个样本

        Uses direct LLM invocation when self.llm is set, otherwise falls back
        to agent.run().  Loads file attachments and includes their content
        in the prompt (text) or as multimodal image blocks (LLM mode only).

        Args:
            agent: 要评估的智能体 (ignored when self.llm is set)
            sample: 样本数据

        Returns:
            单个样本的评估结果
        """
        try:
            # 准备输入
            question = sample.get("question", "")
            expected_answer = sample.get("final_answer", "")
            level = sample.get("level", 1)
            task_id = sample.get("task_id", "")

            # Load file attachment
            file_name = sample.get("file_name", "")
            content_type, file_content = self._load_file_content(file_name) if file_name else (None, None)

            # Skip unsupported file types
            if file_name and content_type is None and file_content is None:
                print(f"   ⏭️ 跳过样本 {task_id}: 不支持的文件类型 ({Path(file_name).suffix})")
                return {
                    "task_id": task_id,
                    "level": level,
                    "exact_match": False,
                    "partial_match": False,
                    "score": 0.0,
                    "predicted": None,
                    "expected": expected_answer,
                    "skipped": True,
                    "skip_reason": f"Unsupported file type: {Path(file_name).suffix}",
                }

            # Build prompt with file text content (if applicable)
            file_text = file_content if content_type == "text" else None
            prompt = self._build_prompt(question, sample, file_text=file_text)

            start_time = time.time()

            if self.llm:
                # Direct LLM mode — prepend system prompt if available
                messages = []
                if self.gaia_system_prompt:
                    messages.append({"role": "system", "content": self.gaia_system_prompt})
                if content_type == "image":
                    messages.extend(self._build_multimodal_messages(prompt, file_content, file_name))
                else:
                    messages.append({"role": "user", "content": prompt})
                response = self.llm.invoke(messages)
            else:
                # Agent mode — images not supported (agent.run takes a string)
                if content_type == "image":
                    print(f"   ⏭️ 跳过样本 {task_id}: agent模式不支持图片文件")
                    return {
                        "task_id": task_id,
                        "level": level,
                        "exact_match": False,
                        "partial_match": False,
                        "score": 0.0,
                        "predicted": None,
                        "expected": expected_answer,
                        "skipped": True,
                        "skip_reason": "Image files not supported in agent mode",
                    }
                response = agent.run(prompt)

            execution_time = time.time() - start_time

            # 提取答案
            predicted_answer = self._extract_answer(response)

            # 评估答案
            exact_match = self._check_exact_match(predicted_answer, expected_answer)
            partial_match = self._check_partial_match(predicted_answer, expected_answer)

            # 计算分数
            if exact_match:
                score = 1.0
            elif partial_match:
                score = 0.5
            else:
                score = 0.0

            return {
                "task_id": task_id,
                "level": level,
                "exact_match": exact_match,
                "partial_match": partial_match,
                "score": score,
                "predicted": predicted_answer,
                "expected": expected_answer,
                "response": response,
                "execution_time": execution_time
            }

        except Exception as e:
            return {
                "task_id": sample.get("task_id", ""),
                "level": sample.get("level", 1),
                "exact_match": False,
                "partial_match": False,
                "score": 0.0,
                "predicted": None,
                "expected": sample.get("final_answer", ""),
                "error": str(e)
            }

    def _create_empty_results(self, agent: Any) -> Dict[str, Any]:
        """创建空的评估结果"""
        agent_name = getattr(self.llm, 'model', None) or getattr(agent, 'name', 'Unknown')
        return {
            "benchmark": "GAIA",
            "agent_name": agent_name,
            "strict_mode": self.strict_mode,
            "level_filter": self.level,
            "total_samples": 0,
            "skipped_samples": 0,
            "exact_matches": 0,
            "partial_matches": 0,
            "exact_match_rate": 0.0,
            "partial_match_rate": 0.0,
            "level_metrics": {},
            "detailed_results": []
        }

    def _build_prompt(self, question: str, sample: Dict[str, Any], file_text: Optional[str] = None) -> str:
        """构建评估提示

        Uses the gaia_task.prompt template if available, otherwise falls back
        to the original inline format.

        Args:
            question: 问题文本
            sample: 样本数据
            file_text: 已加载的文件文本内容 (for text-type attachments)
        """
        if self.task_template:
            file_section = ""
            if file_text and sample.get("file_name"):
                file_basename = Path(sample["file_name"]).name
                file_section = f"\n## Attached File: {file_basename}\n\n{file_text}\n"
            elif sample.get("file_name") and not file_text:
                file_section = f"\nNote: This question may require reference to the file: {Path(sample['file_name']).name}\n"
            return self.task_template.format(
                question=question,
                file_section=file_section,
            )

        # Fallback: original inline prompt
        prompt = f"{question}"

        if file_text and sample.get("file_name"):
            file_basename = Path(sample["file_name"]).name
            prompt += f"\n\n## Attached File: {file_basename}\n\n{file_text}"
        elif sample.get("file_name") and not file_text:
            prompt += f"\n\nNote: This question may require reference to the file: {Path(sample['file_name']).name}"

        return prompt

    def _load_file_content(self, file_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Read a file from disk and return its content for prompt inclusion.

        Returns:
            (content_type, content) where content_type is "text", "image", or
            None (unsupported).  For text, content is the string.  For image,
            content is a base64 data-URL.
        """
        if not file_path:
            return None, None

        path = Path(file_path)
        if not path.exists():
            print(f"   ⚠️ 文件不存在: {file_path}")
            return None, None

        suffix = path.suffix.lower()

        # --- Text-readable formats ---
        text_extensions = {
            ".txt", ".md", ".py", ".csv", ".json", ".jsonl",
            ".xml", ".html", ".log", ".tsv", ".yaml", ".yml",
        }
        if suffix in text_extensions:
            try:
                content = path.read_text(encoding="utf-8")
                # Truncate very large files to avoid blowing up the context
                max_chars = 50_000
                if len(content) > max_chars:
                    content = content[:max_chars] + "\n\n... [truncated]"
                return "text", content
            except Exception as e:
                print(f"   ⚠️ 文件读取失败 ({path.name}): {e}")
                return None, None

        # --- Excel ---
        if suffix in (".xlsx", ".xls"):
            try:
                import openpyxl
                wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
                text_parts = []
                for sheet_name in wb.sheetnames:
                    ws = wb[sheet_name]
                    text_parts.append(f"### Sheet: {sheet_name}\n")
                    for row in ws.iter_rows(values_only=True):
                        row_str = "\t".join(str(cell) if cell is not None else "" for cell in row)
                        text_parts.append(row_str)
                wb.close()
                content = "\n".join(text_parts)
                max_chars = 50_000
                if len(content) > max_chars:
                    content = content[:max_chars] + "\n\n... [truncated]"
                return "text", content
            except ImportError:
                print("   ⚠️ openpyxl未安装, 无法读取Excel文件")
                return None, None
            except Exception as e:
                print(f"   ⚠️ Excel读取失败 ({path.name}): {e}")
                return None, None

        # --- PDF ---
        if suffix == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(str(path))
                pages_text = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        pages_text.append(text)
                content = "\n\n".join(pages_text)
                max_chars = 50_000
                if len(content) > max_chars:
                    content = content[:max_chars] + "\n\n... [truncated]"
                return "text", content
            except ImportError:
                try:
                    import PyPDF2
                    reader = PyPDF2.PdfReader(str(path))
                    pages_text = []
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            pages_text.append(text)
                    content = "\n\n".join(pages_text)
                    max_chars = 50_000
                    if len(content) > max_chars:
                        content = content[:max_chars] + "\n\n... [truncated]"
                    return "text", content
                except ImportError:
                    print("   ⚠️ pypdf/PyPDF2未安装, 无法读取PDF文件")
                    return None, None
            except Exception as e:
                print(f"   ⚠️ PDF读取失败 ({path.name}): {e}")
                return None, None

        # --- Image ---
        image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
        if suffix in image_extensions:
            try:
                mime_map = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                    ".bmp": "image/bmp",
                }
                mime = mime_map.get(suffix, "image/png")
                raw = path.read_bytes()
                b64 = base64.b64encode(raw).decode("ascii")
                data_url = f"data:{mime};base64,{b64}"
                return "image", data_url
            except Exception as e:
                print(f"   ⚠️ 图片读取失败 ({path.name}): {e}")
                return None, None

        # --- Unsupported ---
        print(f"   ⚠️ 不支持的文件类型: {suffix}")
        return None, None

    def _build_multimodal_messages(self, prompt: str, image_data_url: str, file_name: str) -> List[Dict]:
        """Construct OpenAI-compatible multimodal messages with an image.

        Args:
            prompt: The text prompt.
            image_data_url: Base64 data URL for the image.
            file_name: Original file name (for logging context).

        Returns:
            Messages list suitable for HelloAgentsLLM.invoke().
        """
        return [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]
        }]

    def _extract_answer(self, response: str) -> str:
        """从响应中提取答案（GAIA格式）

        GAIA要求答案格式为：FINAL ANSWER: [答案]
        """
        # 首先尝试提取GAIA官方格式的答案
        final_answer_pattern = r'FINAL ANSWER:\s*(.+?)(?:\n|$)'
        match = re.search(final_answer_pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            answer = match.group(1).strip()
            # 移除可能的方括号
            answer = answer.strip('[]')
            return answer

        # 备用方案：查找其他答案标记
        answer_patterns = [
            r'答案[：:]\s*(.+)',
            r'最终答案[：:]\s*(.+)',
            r'Final answer[：:]\s*(.+)',
            r'Answer[：:]\s*(.+)',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # 如果没有找到标记，返回最后一个非空行
        lines = response.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('#'):
                return line

        return response.strip()

    def _check_exact_match(self, predicted: str, expected: str) -> bool:
        """检查精确匹配"""
        if not predicted or not expected:
            return False

        # 标准化字符串
        pred_normalized = self._normalize_answer(predicted)
        exp_normalized = self._normalize_answer(expected)

        return pred_normalized == exp_normalized

    def _check_partial_match(self, predicted: str, expected: str) -> bool:
        """检查部分匹配"""
        if not predicted or not expected:
            return False

        # 标准化字符串
        pred_normalized = self._normalize_answer(predicted)
        exp_normalized = self._normalize_answer(expected)

        # 检查包含关系
        if exp_normalized in pred_normalized or pred_normalized in exp_normalized:
            return True

        # 检查关键词匹配
        pred_words = set(pred_normalized.split())
        exp_words = set(exp_normalized.split())

        if not exp_words:
            return False

        # 如果超过70%的期望词汇出现在预测中，认为部分匹配
        overlap = len(pred_words & exp_words)
        return overlap / len(exp_words) >= 0.7

    def _normalize_answer(self, answer: str) -> str:
        """标准化答案字符串（GAIA官方标准化规则）

        根据GAIA论文的标准化规则：
        1. 数字：移除逗号分隔符和单位符号
        2. 字符串：移除冠词、转小写、移除多余空格
        3. 列表：按逗号分隔，每个元素独立标准化
        """
        if not answer:
            return ""

        answer = answer.strip()

        # 检查是否是逗号分隔的列表
        if ',' in answer:
            # 分隔并标准化每个元素
            parts = [self._normalize_single_answer(p.strip()) for p in answer.split(',')]
            # 按字母顺序排序（GAIA要求）
            parts.sort()
            return ','.join(parts)
        else:
            return self._normalize_single_answer(answer)

    def _normalize_single_answer(self, answer: str) -> str:
        """标准化单个答案（不包含逗号的答案）"""
        answer = answer.strip().lower()

        # 移除常见的冠词
        articles = ['the', 'a', 'an']
        words = answer.split()
        if words and words[0] in articles:
            words = words[1:]
            answer = ' '.join(words)

        # 移除货币符号和百分号
        answer = answer.replace('$', '').replace('%', '').replace('€', '').replace('£', '')

        # 移除数字中的逗号分隔符（如 1,000 -> 1000）
        # 但保留小数点
        answer = re.sub(r'(\d),(\d)', r'\1\2', answer)

        # 移除多余空格
        answer = ' '.join(answer.split())

        # 移除末尾的标点符号
        answer = answer.rstrip('.,;:!?')

        return answer

    def export_to_gaia_format(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path],
        include_reasoning: bool = True
    ) -> None:
        """导出为GAIA官方格式

        GAIA格式要求：
        - JSONL格式（每行一个JSON对象）
        - 每个对象包含：task_id, model_answer, reasoning_trace（可选）

        Args:
            results: 评估结果
            output_path: 输出文件路径
            include_reasoning: 是否包含推理轨迹
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        detailed_results = results.get("detailed_results", [])

        with open(output_path, 'w', encoding='utf-8') as f:
            for result in detailed_results:
                gaia_result = {
                    "task_id": result.get("task_id", ""),
                    "model_answer": result.get("predicted", "")
                }

                if include_reasoning:
                    gaia_result["reasoning_trace"] = result.get("response", "")

                f.write(json.dumps(gaia_result, ensure_ascii=False) + '\n')

        print(f"✅ GAIA格式结果已导出")
        print(f"   输出文件: {output_path}")
        print(f"   样本数: {len(detailed_results)}")
        print(f"   包含推理轨迹: {include_reasoning}")

