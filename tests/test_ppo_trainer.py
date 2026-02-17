"""PPOTrainerWrapper 单元测试

测试PPO训练器封装的各个功能，包括：
- 初始化和配置
- 模型加载（策略模型、奖励模型、价值模型、参考模型）
- 数据集准备和tokenization
- 完整训练流程（使用小模型进行端到端验证）

运行方式:
    cd /home/kemove/LLM_Projects/CodingAgent
    python tests/test_ppo_trainer.py
"""

import os
import sys
import shutil
import tempfile
import types
import unittest
from unittest.mock import MagicMock
from pathlib import Path
import importlib

# === 直接导入 rl 子模块，避免触发 code/__init__.py 的全量导入链 ===
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 创建一个占位的 code 包，仅注册 code.rl 子模块
# 避免触发 code/__init__.py 中的全量导入（其中依赖 hello_agents）
_code_pkg = types.ModuleType("code")
_code_pkg.__path__ = [str(project_root / "code")]
_code_pkg.__package__ = "code"
sys.modules.setdefault("code", _code_pkg)

# 创建 code.rl 包占位
_rl_pkg = types.ModuleType("code.rl")
_rl_pkg.__path__ = [str(project_root / "code" / "rl")]
_rl_pkg.__package__ = "code.rl"
sys.modules["code.rl"] = _rl_pkg

# 导入具体模块
_utils_spec = importlib.util.spec_from_file_location(
    "code.rl.utils", str(project_root / "code" / "rl" / "utils.py")
)
rl_utils = importlib.util.module_from_spec(_utils_spec)
sys.modules["code.rl.utils"] = rl_utils
_utils_spec.loader.exec_module(rl_utils)
# 让 .utils 的相对导入正常工作
_rl_pkg.utils = rl_utils

_trainers_spec = importlib.util.spec_from_file_location(
    "code.rl.trainers", str(project_root / "code" / "rl" / "trainers.py")
)
rl_trainers = importlib.util.module_from_spec(_trainers_spec)
sys.modules["code.rl.trainers"] = rl_trainers
_trainers_spec.loader.exec_module(rl_trainers)
_rl_pkg.trainers = rl_trainers

from code.rl.trainers import PPOTrainerWrapper, BaseTrainerWrapper
from code.rl.utils import TrainingConfig


class TestPPOTrainerWrapperInit(unittest.TestCase):
    """测试PPOTrainerWrapper初始化"""

    def test_default_init(self):
        """测试默认初始化"""
        wrapper = PPOTrainerWrapper()
        self.assertIsNotNone(wrapper.config)
        self.assertIsNone(wrapper.dataset)
        self.assertIsNone(wrapper.reward_model)
        self.assertIsNone(wrapper.value_model)
        self.assertIsNone(wrapper.ref_model)
        self.assertIsNone(wrapper.peft_config)
        self.assertIsNone(wrapper.reward_model_name)
        self.assertIsNone(wrapper.value_model_name)
        self.assertIsNone(wrapper.model)
        self.assertIsNone(wrapper.tokenizer)
        self.assertIsNone(wrapper.trainer)

    def test_custom_config(self):
        """测试自定义配置"""
        config = TrainingConfig(
            model_name="Qwen/Qwen3-0.6B",
            learning_rate=1e-5,
            num_train_epochs=2,
            output_dir="/tmp/test_ppo_output",
        )
        wrapper = PPOTrainerWrapper(config=config)
        self.assertEqual(wrapper.config.model_name, "Qwen/Qwen3-0.6B")
        self.assertEqual(wrapper.config.learning_rate, 1e-5)
        self.assertEqual(wrapper.config.num_train_epochs, 2)

    def test_init_with_model_names(self):
        """测试使用模型名称初始化"""
        wrapper = PPOTrainerWrapper(
            reward_model_name="some-reward-model",
            value_model_name="some-value-model",
        )
        self.assertEqual(wrapper.reward_model_name, "some-reward-model")
        self.assertEqual(wrapper.value_model_name, "some-value-model")

    def test_init_with_external_models(self):
        """测试传入外部模型对象"""
        mock_reward = MagicMock()
        mock_value = MagicMock()
        mock_ref = MagicMock()

        wrapper = PPOTrainerWrapper(
            reward_model=mock_reward,
            value_model=mock_value,
            ref_model=mock_ref,
        )
        self.assertIs(wrapper.reward_model, mock_reward)
        self.assertIs(wrapper.value_model, mock_value)
        self.assertIs(wrapper.ref_model, mock_ref)

    def test_inherits_base_trainer(self):
        """确认继承自 BaseTrainerWrapper"""
        self.assertTrue(issubclass(PPOTrainerWrapper, BaseTrainerWrapper))

    def test_has_required_methods(self):
        """确认具有所有必需方法"""
        wrapper = PPOTrainerWrapper()
        self.assertTrue(hasattr(wrapper, "setup_model"))
        self.assertTrue(hasattr(wrapper, "train"))
        self.assertTrue(hasattr(wrapper, "save_model"))
        self.assertTrue(hasattr(wrapper, "prepare_dataset"))
        self.assertTrue(callable(wrapper.setup_model))
        self.assertTrue(callable(wrapper.train))
        self.assertTrue(callable(wrapper.save_model))
        self.assertTrue(callable(wrapper.prepare_dataset))


class TestPPOTrainerWrapperSetupModel(unittest.TestCase):
    """测试PPOTrainerWrapper模型加载"""

    test_model_name = "hf-internal-testing/tiny-random-GPT2LMHeadModel"
    seq_cls_model_name = "hf-internal-testing/tiny-random-GPT2ForSequenceClassification"

    def test_setup_model_loads_all_components(self):
        """测试 setup_model 加载所有组件"""
        config = TrainingConfig(model_name=self.test_model_name)
        wrapper = PPOTrainerWrapper(
            config=config,
            reward_model_name=self.seq_cls_model_name,
            value_model_name=self.seq_cls_model_name,
        )
        wrapper.setup_model()

        self.assertIsNotNone(wrapper.model, "策略模型应已加载")
        self.assertIsNotNone(wrapper.tokenizer, "Tokenizer应已加载")
        self.assertIsNotNone(wrapper.reward_model, "奖励模型应已加载")
        self.assertIsNotNone(wrapper.value_model, "价值模型应已加载")
        self.assertIsNotNone(wrapper.ref_model, "参考模型应已创建")

    def test_setup_model_sets_pad_token(self):
        """测试 pad_token 被正确设置"""
        config = TrainingConfig(model_name=self.test_model_name)
        wrapper = PPOTrainerWrapper(
            config=config,
            reward_model_name=self.seq_cls_model_name,
            value_model_name=self.seq_cls_model_name,
        )
        wrapper.setup_model()
        self.assertIsNotNone(wrapper.tokenizer.pad_token)

    def test_setup_model_skips_ref_when_peft(self):
        """测试使用 PEFT 配置时不创建 ref_model"""
        mock_peft = MagicMock()
        config = TrainingConfig(model_name=self.test_model_name)
        wrapper = PPOTrainerWrapper(
            config=config,
            reward_model_name=self.seq_cls_model_name,
            value_model_name=self.seq_cls_model_name,
            peft_config=mock_peft,
        )
        wrapper.setup_model()
        self.assertIsNone(wrapper.ref_model)

    def test_setup_model_preserves_external_models(self):
        """测试外部传入的模型不会被覆盖"""
        mock_reward = MagicMock()
        mock_value = MagicMock()
        mock_ref = MagicMock()

        config = TrainingConfig(model_name=self.test_model_name)
        wrapper = PPOTrainerWrapper(
            config=config,
            reward_model=mock_reward,
            value_model=mock_value,
            ref_model=mock_ref,
        )
        wrapper.setup_model()

        self.assertIs(wrapper.reward_model, mock_reward)
        self.assertIs(wrapper.value_model, mock_value)
        self.assertIs(wrapper.ref_model, mock_ref)

    def test_ref_model_is_deep_copy(self):
        """测试参考模型是策略模型的独立副本"""
        config = TrainingConfig(model_name=self.test_model_name)
        wrapper = PPOTrainerWrapper(
            config=config,
            reward_model_name=self.seq_cls_model_name,
            value_model_name=self.seq_cls_model_name,
        )
        wrapper.setup_model()

        # ref_model 和 model 不应是同一对象
        self.assertIsNot(wrapper.ref_model, wrapper.model)
        # 但应具有相同的参数
        for (n1, p1), (n2, p2) in zip(
            wrapper.model.named_parameters(), wrapper.ref_model.named_parameters()
        ):
            self.assertEqual(n1, n2)
            self.assertTrue(p1.equal(p2))


class TestPPOTrainerWrapperPrepareDataset(unittest.TestCase):
    """测试PPOTrainerWrapper数据集准备"""

    test_model_name = "hf-internal-testing/tiny-random-GPT2LMHeadModel"

    def _make_wrapper_with_tokenizer(self):
        """创建一个已初始化 tokenizer 的 wrapper"""
        from transformers import AutoTokenizer

        config = TrainingConfig(model_name=self.test_model_name, max_length=64)
        wrapper = PPOTrainerWrapper(config=config)
        wrapper.tokenizer = AutoTokenizer.from_pretrained(
            self.test_model_name, trust_remote_code=True
        )
        if wrapper.tokenizer.pad_token is None:
            wrapper.tokenizer.pad_token = wrapper.tokenizer.eos_token
        return wrapper

    def test_prepare_dataset_with_prompt_column(self):
        """测试从 prompt 列自动 tokenize"""
        from datasets import Dataset

        wrapper = self._make_wrapper_with_tokenizer()
        ds = Dataset.from_dict({
            "prompt": ["Hello world", "How are you?", "Test prompt here"],
        })
        wrapper.dataset = ds
        result = wrapper.prepare_dataset()

        self.assertIn("input_ids", result.column_names)
        self.assertIn("attention_mask", result.column_names)
        self.assertEqual(len(result), 3)

    def test_prepare_dataset_with_existing_input_ids(self):
        """测试已有 input_ids 的数据集不做额外处理"""
        from datasets import Dataset

        wrapper = self._make_wrapper_with_tokenizer()
        ds = Dataset.from_dict({
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "attention_mask": [[1, 1, 1], [1, 1, 1]],
        })
        wrapper.dataset = ds
        result = wrapper.prepare_dataset()

        self.assertEqual(len(result), 2)
        self.assertIn("input_ids", result.column_names)

    def test_prepare_dataset_raises_without_dataset(self):
        """测试未设置数据集时抛出异常"""
        wrapper = self._make_wrapper_with_tokenizer()
        wrapper.dataset = None
        with self.assertRaises(ValueError):
            wrapper.prepare_dataset()

    def test_prepare_dataset_raises_without_prompt_or_input_ids(self):
        """测试数据集缺少必要列时抛出异常"""
        from datasets import Dataset

        wrapper = self._make_wrapper_with_tokenizer()
        ds = Dataset.from_dict({"text": ["some text"]})
        wrapper.dataset = ds
        with self.assertRaises(ValueError):
            wrapper.prepare_dataset()

    def test_prepare_dataset_raises_without_tokenizer(self):
        """测试tokenizer未初始化时抛出异常"""
        from datasets import Dataset

        config = TrainingConfig(model_name=self.test_model_name)
        wrapper = PPOTrainerWrapper(config=config)
        wrapper.tokenizer = None

        ds = Dataset.from_dict({"prompt": ["Hello"]})
        wrapper.dataset = ds
        with self.assertRaises(ValueError):
            wrapper.prepare_dataset()

    def test_prepare_dataset_accepts_external_dataset(self):
        """测试使用外部传入的数据集参数"""
        from datasets import Dataset

        wrapper = self._make_wrapper_with_tokenizer()
        wrapper.dataset = Dataset.from_dict({"prompt": ["dataset A"]})

        external_ds = Dataset.from_dict({"prompt": ["dataset B"]})
        result = wrapper.prepare_dataset(dataset=external_ds)

        self.assertEqual(len(result), 1)
        self.assertIn("input_ids", result.column_names)


class TestPPOTrainerWrapperTrainValidation(unittest.TestCase):
    """测试PPOTrainerWrapper训练前验证"""

    def test_train_raises_without_dataset(self):
        """测试未设置数据集时训练抛出异常"""
        wrapper = PPOTrainerWrapper(dataset=None)
        wrapper.model = MagicMock()
        wrapper.tokenizer = MagicMock()

        with self.assertRaises(ValueError):
            wrapper.train()


class TestPPOTrainerWrapperSaveModel(unittest.TestCase):
    """测试模型保存功能"""

    def test_save_model_without_trainer(self):
        """测试 trainer 未初始化时保存模型的行为"""
        wrapper = PPOTrainerWrapper()
        # 不应抛出异常，只是打印警告
        wrapper.save_model("/tmp/nonexistent")

    def test_save_model_with_trainer(self):
        """测试 trainer 已初始化时保存模型"""
        wrapper = PPOTrainerWrapper()
        wrapper.trainer = MagicMock()

        output_dir = tempfile.mkdtemp(prefix="test_save_")
        try:
            wrapper.save_model(output_dir)
            wrapper.trainer.save_model.assert_called_once_with(output_dir)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


class TestPPOTrainerWrapperE2E(unittest.TestCase):
    """端到端集成测试 - 使用小模型验证完整训练流程

    此测试使用 hf-internal-testing 的微型模型，确保能在 CPU 上快速运行。
    """

    test_model_name = "hf-internal-testing/tiny-random-GPT2LMHeadModel"
    seq_cls_model_name = "hf-internal-testing/tiny-random-GPT2ForSequenceClassification"

    @classmethod
    def setUpClass(cls):
        cls.output_dir = tempfile.mkdtemp(prefix="test_ppo_e2e_")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.output_dir):
            shutil.rmtree(cls.output_dir, ignore_errors=True)

    def test_full_training_pipeline(self):
        """端到端测试：从模型加载到训练完成（使用预 tokenized 数据集）"""
        from datasets import Dataset
        from transformers import AutoTokenizer

        config = TrainingConfig(
            model_name=self.test_model_name,
            output_dir=os.path.join(self.output_dir, "e2e_test"),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            logging_steps=1,
            save_steps=999999,
            max_length=32,
            max_new_tokens=16,
            use_fp16=False,
            use_bf16=False,
            use_wandb=False,
            use_tensorboard=False,
        )

        # 准备 tokenized 数据集
        tokenizer = AutoTokenizer.from_pretrained(
            self.test_model_name, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        prompts = ["Hello world", "How are you", "Test input", "Another test"]
        encodings = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors=None,
        )
        dataset = Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        })
        dataset.set_format(type="torch")

        wrapper = PPOTrainerWrapper(
            config=config,
            dataset=dataset,
            reward_model_name=self.seq_cls_model_name,
            value_model_name=self.seq_cls_model_name,
        )

        trainer = wrapper.train()

        self.assertIsNotNone(trainer)
        self.assertIsNotNone(wrapper.trainer)

    def test_training_with_prompt_dataset(self):
        """端到端测试：使用 prompt 格式的数据集（自动 tokenize）"""
        from datasets import Dataset

        config = TrainingConfig(
            model_name=self.test_model_name,
            output_dir=os.path.join(self.output_dir, "prompt_test"),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            logging_steps=1,
            save_steps=999999,
            max_length=32,
            max_new_tokens=16,
            use_fp16=False,
            use_bf16=False,
            use_wandb=False,
            use_tensorboard=False,
        )

        dataset = Dataset.from_dict({
            "prompt": ["Hello world", "Test prompt", "Another one", "Last one"],
        })

        wrapper = PPOTrainerWrapper(
            config=config,
            dataset=dataset,
            reward_model_name=self.seq_cls_model_name,
            value_model_name=self.seq_cls_model_name,
        )

        trainer = wrapper.train()
        self.assertIsNotNone(trainer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
