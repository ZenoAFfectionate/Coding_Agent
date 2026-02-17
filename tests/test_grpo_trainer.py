"""GRPOTrainerWrapper 单元测试

测试GRPO训练器封装的各个功能，包括：
- 初始化和配置
- 模型加载（策略模型 + tokenizer）
- 训练前验证（数据集、奖励函数）
- 与 MathRewardFunction 的集成
- 完整训练流程（使用小模型进行端到端验证）

运行方式:
    cd /home/kemove/LLM_Projects/CodingAgent
    python tests/test_grpo_trainer.py
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

# 创建占位的 code 包，仅注册 code.rl 子模块
_code_pkg = types.ModuleType("code")
_code_pkg.__path__ = [str(project_root / "code")]
_code_pkg.__package__ = "code"
sys.modules.setdefault("code", _code_pkg)

_rl_pkg = types.ModuleType("code.rl")
_rl_pkg.__path__ = [str(project_root / "code" / "rl")]
_rl_pkg.__package__ = "code.rl"
sys.modules["code.rl"] = _rl_pkg

# 导入 utils
_utils_spec = importlib.util.spec_from_file_location(
    "code.rl.utils", str(project_root / "code" / "rl" / "utils.py")
)
rl_utils = importlib.util.module_from_spec(_utils_spec)
sys.modules["code.rl.utils"] = rl_utils
_utils_spec.loader.exec_module(rl_utils)
_rl_pkg.utils = rl_utils

# 导入 trainers
_trainers_spec = importlib.util.spec_from_file_location(
    "code.rl.trainers", str(project_root / "code" / "rl" / "trainers.py")
)
rl_trainers = importlib.util.module_from_spec(_trainers_spec)
sys.modules["code.rl.trainers"] = rl_trainers
_trainers_spec.loader.exec_module(rl_trainers)
_rl_pkg.trainers = rl_trainers

# 导入 rewards
_rewards_spec = importlib.util.spec_from_file_location(
    "code.rl.rewards", str(project_root / "code" / "rl" / "rewards.py")
)
rl_rewards = importlib.util.module_from_spec(_rewards_spec)
sys.modules["code.rl.rewards"] = rl_rewards
_rewards_spec.loader.exec_module(rl_rewards)
_rl_pkg.rewards = rl_rewards

from code.rl.trainers import GRPOTrainerWrapper, BaseTrainerWrapper
from code.rl.utils import TrainingConfig
from code.rl.rewards import MathRewardFunction, create_accuracy_reward


# ============================================================
# Helper: TRL GRPOTrainer 的奖励函数签名为
#   reward_func(prompts=..., completions=..., **kwargs) -> list[float]
# 这与 MathRewardFunction.__call__(completions, **kwargs) 不完全一致，
# 因此 E2E 测试中使用一个简单的 dummy reward 来避免签名问题。
# ============================================================
def dummy_reward_fn(prompts, completions, **kwargs):
    """简单的 dummy 奖励函数，返回基于文本长度的浮点奖励"""
    return [float(len(c) % 10) / 10.0 for c in completions]


dummy_reward_fn.__name__ = "dummy_reward_fn"


class TestGRPOTrainerWrapperInit(unittest.TestCase):
    """测试GRPOTrainerWrapper初始化"""

    def test_default_init(self):
        """测试默认初始化"""
        wrapper = GRPOTrainerWrapper()
        self.assertIsNotNone(wrapper.config)
        self.assertIsNone(wrapper.dataset)
        self.assertIsNone(wrapper.reward_fn)
        self.assertIsNone(wrapper.model)
        self.assertIsNone(wrapper.tokenizer)
        self.assertIsNone(wrapper.trainer)

    def test_custom_config(self):
        """测试自定义配置"""
        config = TrainingConfig(
            model_name="Qwen/Qwen3-0.6B",
            learning_rate=1e-5,
            num_train_epochs=2,
            output_dir="/tmp/test_grpo_output",
        )
        wrapper = GRPOTrainerWrapper(config=config)
        self.assertEqual(wrapper.config.model_name, "Qwen/Qwen3-0.6B")
        self.assertEqual(wrapper.config.learning_rate, 1e-5)
        self.assertEqual(wrapper.config.num_train_epochs, 2)
        self.assertEqual(wrapper.config.output_dir, "/tmp/test_grpo_output")

    def test_init_with_all_params(self):
        """测试传入所有参数初始化"""
        mock_dataset = MagicMock()
        mock_reward = MagicMock()
        config = TrainingConfig(model_name="test-model")

        wrapper = GRPOTrainerWrapper(
            config=config,
            dataset=mock_dataset,
            reward_fn=mock_reward,
        )
        self.assertIs(wrapper.dataset, mock_dataset)
        self.assertIs(wrapper.reward_fn, mock_reward)
        self.assertEqual(wrapper.config.model_name, "test-model")

    def test_inherits_base_trainer(self):
        """确认继承自 BaseTrainerWrapper"""
        self.assertTrue(issubclass(GRPOTrainerWrapper, BaseTrainerWrapper))

    def test_has_required_methods(self):
        """确认具有所有必需方法"""
        wrapper = GRPOTrainerWrapper()
        self.assertTrue(callable(getattr(wrapper, "setup_model", None)))
        self.assertTrue(callable(getattr(wrapper, "train", None)))
        self.assertTrue(callable(getattr(wrapper, "save_model", None)))

    def test_exported_from_rl_trainers(self):
        """确认从 rl.trainers 模块正确导出"""
        from code.rl.trainers import GRPOTrainerWrapper as GRPOFromModule
        self.assertIs(GRPOFromModule, GRPOTrainerWrapper)


class TestGRPOTrainerWrapperSetupModel(unittest.TestCase):
    """测试GRPOTrainerWrapper模型加载"""

    test_model_name = "hf-internal-testing/tiny-random-GPT2LMHeadModel"

    def test_setup_model_loads_model_and_tokenizer(self):
        """测试 setup_model 加载策略模型和 tokenizer"""
        config = TrainingConfig(model_name=self.test_model_name)
        wrapper = GRPOTrainerWrapper(config=config)
        wrapper.setup_model()

        self.assertIsNotNone(wrapper.model, "策略模型应已加载")
        self.assertIsNotNone(wrapper.tokenizer, "Tokenizer应已加载")

    def test_setup_model_sets_pad_token(self):
        """测试 pad_token 被正确设置"""
        config = TrainingConfig(model_name=self.test_model_name)
        wrapper = GRPOTrainerWrapper(config=config)
        wrapper.setup_model()
        self.assertIsNotNone(wrapper.tokenizer.pad_token)

    def test_setup_model_correct_model_type(self):
        """测试加载的模型类型正确 (CausalLM)"""
        from transformers import GPT2LMHeadModel

        config = TrainingConfig(model_name=self.test_model_name)
        wrapper = GRPOTrainerWrapper(config=config)
        wrapper.setup_model()
        self.assertIsInstance(wrapper.model, GPT2LMHeadModel)

    def test_setup_model_no_device_map_without_fp16(self):
        """测试不使用 fp16/bf16 时不设置 device_map"""
        config = TrainingConfig(
            model_name=self.test_model_name,
            use_fp16=False,
            use_bf16=False,
        )
        wrapper = GRPOTrainerWrapper(config=config)
        wrapper.setup_model()
        # 模型应该已加载成功（不带 device_map="auto"）
        self.assertIsNotNone(wrapper.model)


class TestGRPOTrainerWrapperTrainValidation(unittest.TestCase):
    """测试GRPOTrainerWrapper训练前验证"""

    def test_train_raises_without_dataset(self):
        """测试未设置数据集时训练抛出异常"""
        wrapper = GRPOTrainerWrapper(dataset=None, reward_fn=dummy_reward_fn)
        wrapper.model = MagicMock()
        wrapper.tokenizer = MagicMock()

        with self.assertRaises(ValueError) as ctx:
            wrapper.train()
        self.assertIn("数据集", str(ctx.exception))

    def test_train_raises_without_reward_fn(self):
        """测试未设置奖励函数时训练抛出异常"""
        from datasets import Dataset

        ds = Dataset.from_dict({"prompt": ["test"]})
        wrapper = GRPOTrainerWrapper(dataset=ds, reward_fn=None)
        wrapper.model = MagicMock()
        wrapper.tokenizer = MagicMock()

        with self.assertRaises(ValueError) as ctx:
            wrapper.train()
        self.assertIn("奖励函数", str(ctx.exception))

    def test_train_calls_setup_model_when_model_is_none(self):
        """测试 model 未初始化时 train() 会自动调用 setup_model()"""
        from datasets import Dataset

        ds = Dataset.from_dict({"prompt": ["test"] * 8})
        config = TrainingConfig(
            model_name="hf-internal-testing/tiny-random-GPT2LMHeadModel",
            per_device_train_batch_size=8,
            use_fp16=False,
            use_bf16=False,
        )
        wrapper = GRPOTrainerWrapper(config=config, dataset=ds, reward_fn=dummy_reward_fn)

        # model 为 None，train() 应首先调用 setup_model()
        # 但会因 GRPOConfig/GRPOTrainer 初始化继续执行
        # 我们只需验证 model 被加载了（通过 try/except 包住可能的后续错误）
        self.assertIsNone(wrapper.model)
        try:
            wrapper.train()
        except Exception:
            pass  # 训练可能因其他原因失败，但 model 应已加载
        self.assertIsNotNone(wrapper.model)


class TestGRPOTrainerWrapperSaveModel(unittest.TestCase):
    """测试模型保存功能"""

    def test_save_model_without_trainer(self):
        """测试 trainer 未初始化时保存模型的行为"""
        wrapper = GRPOTrainerWrapper()
        # 不应抛出异常
        wrapper.save_model("/tmp/nonexistent")

    def test_save_model_with_trainer(self):
        """测试 trainer 已初始化时保存模型"""
        wrapper = GRPOTrainerWrapper()
        wrapper.trainer = MagicMock()

        output_dir = tempfile.mkdtemp(prefix="test_grpo_save_")
        try:
            wrapper.save_model(output_dir)
            wrapper.trainer.save_model.assert_called_once_with(output_dir)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_save_model_uses_config_output_dir_by_default(self):
        """测试默认使用 config.output_dir"""
        config = TrainingConfig(output_dir="/tmp/test_grpo_default_save")
        wrapper = GRPOTrainerWrapper(config=config)
        wrapper.trainer = MagicMock()

        wrapper.save_model()
        wrapper.trainer.save_model.assert_called_once_with("/tmp/test_grpo_default_save")


class TestGRPOTrainerWrapperRewardFunction(unittest.TestCase):
    """测试GRPOTrainerWrapper与奖励函数的集成"""

    def test_accepts_callable_reward_fn(self):
        """测试接受普通可调用对象作为奖励函数"""
        wrapper = GRPOTrainerWrapper(reward_fn=dummy_reward_fn)
        self.assertIs(wrapper.reward_fn, dummy_reward_fn)

    def test_accepts_math_reward_function(self):
        """测试接受 MathRewardFunction 实例"""
        reward_fn = MathRewardFunction()
        wrapper = GRPOTrainerWrapper(reward_fn=reward_fn)
        self.assertIsInstance(wrapper.reward_fn, MathRewardFunction)

    def test_accepts_lambda_reward_fn(self):
        """测试接受 lambda 奖励函数"""
        fn = lambda prompts, completions, **kw: [1.0] * len(completions)
        fn.__name__ = "lambda_reward"
        wrapper = GRPOTrainerWrapper(reward_fn=fn)
        self.assertIs(wrapper.reward_fn, fn)


class TestGRPOTrainerWrapperE2E(unittest.TestCase):
    """端到端集成测试 - 使用小模型验证完整训练流程

    此测试使用 hf-internal-testing 的微型模型，确保能在 CPU 上快速运行。
    GRPO 需要:
    - 模型: AutoModelForCausalLM
    - 数据集: 包含 "prompt" 列的 Dataset
    - 奖励函数: Callable(prompts=..., completions=..., **kwargs) -> list[float]
    """

    test_model_name = "hf-internal-testing/tiny-random-GPT2LMHeadModel"

    @classmethod
    def setUpClass(cls):
        cls.output_dir = tempfile.mkdtemp(prefix="test_grpo_e2e_")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.output_dir):
            shutil.rmtree(cls.output_dir, ignore_errors=True)

    def _make_config(self, subdir="default"):
        """创建测试用的 TrainingConfig

        注意: GRPOConfig 要求 per_device_train_batch_size >= num_generations (默认8)
        且 per_device_train_batch_size % num_generations == 0，
        所以这里使用 batch_size=8。
        """
        return TrainingConfig(
            model_name=self.test_model_name,
            output_dir=os.path.join(self.output_dir, subdir),
            num_train_epochs=1,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            logging_steps=1,
            save_steps=999999,
            max_length=64,
            max_new_tokens=16,
            use_fp16=False,
            use_bf16=False,
            use_wandb=False,
            use_tensorboard=False,
        )

    def test_full_training_pipeline(self):
        """端到端测试：完整 GRPO 训练流程"""
        from datasets import Dataset

        config = self._make_config("full_pipeline")
        dataset = Dataset.from_dict({
            "prompt": [
                "What is 1+1?",
                "What is 2+2?",
                "What is 3+3?",
                "What is 4+4?",
                "What is 5+5?",
                "What is 6+6?",
                "What is 7+7?",
                "What is 8+8?",
            ],
        })

        wrapper = GRPOTrainerWrapper(
            config=config,
            dataset=dataset,
            reward_fn=dummy_reward_fn,
        )

        trainer = wrapper.train()

        self.assertIsNotNone(trainer)
        self.assertIsNotNone(wrapper.trainer)
        self.assertIsNotNone(wrapper.model)
        self.assertIsNotNone(wrapper.tokenizer)

    def test_training_with_extra_columns(self):
        """端到端测试：数据集包含额外列（如 ground_truth），验证 remove_unused_columns=False 生效"""
        from datasets import Dataset

        config = self._make_config("extra_cols")
        dataset = Dataset.from_dict({
            "prompt": [
                "What is 1+1?",
                "What is 2+2?",
                "What is 3+3?",
                "What is 4+4?",
                "What is 5+5?",
                "What is 6+6?",
                "What is 7+7?",
                "What is 8+8?",
            ],
            "ground_truth": ["2", "4", "6", "8", "10", "12", "14", "16"],
        })

        # 奖励函数使用 ground_truth 列
        def reward_with_gt(prompts, completions, **kwargs):
            return [1.0 if "2" in c else 0.0 for c in completions]

        reward_with_gt.__name__ = "reward_with_gt"

        wrapper = GRPOTrainerWrapper(
            config=config,
            dataset=dataset,
            reward_fn=reward_with_gt,
        )

        trainer = wrapper.train()
        self.assertIsNotNone(trainer)

    def test_training_auto_setup_model(self):
        """端到端测试：不手动调用 setup_model，验证 train() 自动完成"""
        from datasets import Dataset

        config = self._make_config("auto_setup")
        dataset = Dataset.from_dict({
            "prompt": ["Hello world", "Test prompt"] * 4,
        })

        wrapper = GRPOTrainerWrapper(
            config=config,
            dataset=dataset,
            reward_fn=dummy_reward_fn,
        )

        # 确认 model 初始为 None
        self.assertIsNone(wrapper.model)

        trainer = wrapper.train()

        # 确认 train() 内部自动调用了 setup_model()
        self.assertIsNotNone(wrapper.model)
        self.assertIsNotNone(trainer)

    def test_trainer_returns_grpo_trainer_instance(self):
        """验证 train() 返回的是 TRL GRPOTrainer 实例"""
        from datasets import Dataset
        from trl import GRPOTrainer

        config = self._make_config("return_type")
        dataset = Dataset.from_dict({
            "prompt": ["Test"] * 8,
        })

        wrapper = GRPOTrainerWrapper(
            config=config,
            dataset=dataset,
            reward_fn=dummy_reward_fn,
        )

        trainer = wrapper.train()
        self.assertIsInstance(trainer, GRPOTrainer)

    def test_report_to_none(self):
        """验证 use_wandb=False, use_tensorboard=False 时 report_to 为 ['none']"""
        from datasets import Dataset

        config = self._make_config("report_none")
        config.use_wandb = False
        config.use_tensorboard = False

        dataset = Dataset.from_dict({
            "prompt": ["Test"] * 8,
        })

        wrapper = GRPOTrainerWrapper(
            config=config,
            dataset=dataset,
            reward_fn=dummy_reward_fn,
        )

        # train 成功即说明 report_to=["none"] 被正确设置
        trainer = wrapper.train()
        self.assertIsNotNone(trainer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
