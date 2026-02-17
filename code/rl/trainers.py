"""RL训练器封装

本模块封装了TRL的各种训练器，提供统一的接口。
"""

from typing import Optional, Callable, Dict, Any
from pathlib import Path

from .utils import TrainingConfig, check_trl_installation, get_installation_guide

try:
    from transformers import TrainerCallback

    class DetailedLoggingCallback(TrainerCallback):
        """详细日志回调

        在训练过程中输出更详细的日志信息,包括:
        - Epoch/Step进度
        - Loss
        - Learning Rate
        - Reward (GRPO)
        - KL散度 (GRPO)
        """

        def __init__(self, total_steps: int = None, num_epochs: int = None):
            """
            初始化回调

            Args:
                total_steps: 总步数
                num_epochs: 总轮数
            """
            self.total_steps = total_steps
            self.num_epochs = num_epochs
            self.current_epoch = 0

        def on_log(self, args, state, control, logs=None, **kwargs):
            """日志回调"""
            if logs is None:
                return

            # 计算当前epoch
            if state.epoch is not None:
                self.current_epoch = int(state.epoch)

            # 构建日志消息
            log_parts = []

            # Epoch和Step信息
            if self.num_epochs:
                log_parts.append(f"Epoch {self.current_epoch + 1}/{self.num_epochs}")

            if state.global_step and self.total_steps:
                log_parts.append(f"Step {state.global_step}/{self.total_steps}")
            elif state.global_step:
                log_parts.append(f"Step {state.global_step}")

            # Loss
            if "loss" in logs:
                log_parts.append(f"Loss: {logs['loss']:.4f}")

            # Learning Rate
            if "learning_rate" in logs:
                log_parts.append(f"LR: {logs['learning_rate']:.2e}")

            # GRPO特定指标
            if "rewards/mean" in logs:
                log_parts.append(f"Reward: {logs['rewards/mean']:.4f}")

            if "objective/kl" in logs:
                log_parts.append(f"KL: {logs['objective/kl']:.4f}")

            # 输出日志
            if log_parts:
                print(" | ".join(log_parts))

        def on_epoch_end(self, args, state, control, **kwargs):
            """Epoch结束回调"""
            print(f"{'='*80}")
            print(f"✅ Epoch {self.current_epoch + 1} 完成")
            print(f"{'='*80}\n")

except ImportError:
    # 如果transformers未安装,创建一个空的回调类
    class DetailedLoggingCallback:
        def __init__(self, *args, **kwargs):
            pass


class BaseTrainerWrapper:
    """训练器基类"""
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        初始化训练器
        
        Args:
            config: 训练配置
        """
        # 检查TRL是否安装
        if not check_trl_installation():
            raise ImportError(get_installation_guide())
        
        self.config = config or TrainingConfig()
        self.trainer = None
        self.model = None
        self.tokenizer = None
    
    def setup_model(self):
        """设置模型和tokenizer"""
        raise NotImplementedError
    
    def train(self):
        """开始训练"""
        raise NotImplementedError
    
    def save_model(self, output_dir: Optional[str] = None):
        """
        保存模型
        
        Args:
            output_dir: 输出目录
        """
        save_dir = output_dir or self.config.output_dir
        if self.trainer:
            self.trainer.save_model(save_dir)
            print(f"✅ 模型已保存到: {save_dir}")
        else:
            print("❌ 训练器未初始化，无法保存模型")


class SFTTrainerWrapper(BaseTrainerWrapper):
    """SFT (Supervised Fine-Tuning) 训练器封装
    
    用于监督微调，让模型学会遵循指令和基本的推理格式。
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        dataset = None
    ):
        """
        初始化SFT训练器
        
        Args:
            config: 训练配置
            dataset: 训练数据集
        """
        super().__init__(config)
        self.dataset = dataset
    
    def setup_model(self):
        """设置模型和tokenizer"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"📦 加载模型: {self.config.model_name}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            device_map="auto" if self.config.use_fp16 or self.config.use_bf16 else None
        )
        
        print("✅ 模型加载完成")
    
    def train(self):
        """开始SFT训练"""
        from trl import SFTConfig, SFTTrainer
        
        if self.model is None:
            self.setup_model()
        
        if self.dataset is None:
            raise ValueError("数据集未设置，请提供训练数据集")
        
        # 配置训练参数
        # 确定report_to参数
        report_to = []
        if self.config.use_wandb:
            report_to.append("wandb")
        if self.config.use_tensorboard:
            report_to.append("tensorboard")
        if not report_to:
            report_to = ["none"]

        training_args = SFTConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            fp16=self.config.use_fp16,
            bf16=self.config.use_bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            max_length=self.config.max_length,  # 修正参数名
            report_to=report_to,
        )
        
        # 计算总步数
        total_steps = (
            len(self.dataset) //
            (self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps)
        ) * self.config.num_train_epochs

        # 创建详细日志回调
        logging_callback = DetailedLoggingCallback(
            total_steps=total_steps,
            num_epochs=self.config.num_train_epochs
        )

        # 创建训练器
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            processing_class=self.tokenizer,  # 新版TRL使用processing_class
            callbacks=[logging_callback],  # 添加回调
        )

        print("\n🚀 开始SFT训练...")
        print(f"{'='*80}\n")
        self.trainer.train()
        print(f"\n{'='*80}")
        print("✅ SFT训练完成")
        
        return self.trainer


class GRPOTrainerWrapper(BaseTrainerWrapper):
    """GRPO (Group Relative Policy Optimization) 训练器封装
    
    用于强化学习训练，优化模型的推理能力。
    GRPO相比PPO更简单，不需要Value Model。
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        dataset = None,
        reward_fn: Optional[Callable] = None
    ):
        """
        初始化GRPO训练器
        
        Args:
            config: 训练配置
            dataset: 训练数据集
            reward_fn: 奖励函数
        """
        super().__init__(config)
        self.dataset = dataset
        self.reward_fn = reward_fn
    
    def setup_model(self):
        """设置模型和tokenizer"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"📦 加载模型: {self.config.model_name}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            device_map="auto" if self.config.use_fp16 or self.config.use_bf16 else None
        )
        
        print("✅ 模型加载完成")
    
    def train(self):
        """开始GRPO训练"""
        from trl import GRPOConfig, GRPOTrainer
        
        if self.model is None:
            self.setup_model()
        
        if self.dataset is None:
            raise ValueError("数据集未设置，请提供训练数据集")
        
        if self.reward_fn is None:
            raise ValueError("奖励函数未设置，请提供reward_fn")
        
        # 确定report_to参数
        report_to = []
        if self.config.use_wandb:
            report_to.append("wandb")
        if self.config.use_tensorboard:
            report_to.append("tensorboard")
        if not report_to:
            report_to = ["none"]

        # 配置训练参数
        training_args = GRPOConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            fp16=self.config.use_fp16,
            bf16=self.config.use_bf16,
            report_to=report_to,
            remove_unused_columns=False,  # 保留所有列,包括ground_truth等
        )
        
        # 计算总步数
        total_steps = (
            len(self.dataset) //
            (self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps)
        ) * self.config.num_train_epochs

        # 创建详细日志回调
        logging_callback = DetailedLoggingCallback(
            total_steps=total_steps,
            num_epochs=self.config.num_train_epochs
        )

        # 创建训练器
        self.trainer = GRPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            reward_funcs=self.reward_fn,
            processing_class=self.tokenizer,
            callbacks=[logging_callback],  # 添加回调
        )

        print("\n🚀 开始GRPO训练...")
        print(f"{'='*80}\n")
        self.trainer.train()
        print(f"\n{'='*80}")
        print("✅ GRPO训练完成")
        
        return self.trainer


class PPOTrainerWrapper(BaseTrainerWrapper):
    """PPO (Proximal Policy Optimization) 训练器封装

    用于强化学习训练，是经典的RL算法。
    相比GRPO，PPO需要额外的Value Model和Reward Model (nn.Module)，但可能获得更好的性能。

    注意：TRL >= 0.24 中 PPOTrainer 已被标记为实验性功能（candidate for removal），
    官方推荐使用 GRPOTrainer 或 DPOTrainer。如果不需要 Value Model 的优势，
    建议优先使用 GRPOTrainerWrapper。

    关键区别（PPO vs GRPO）：
    - PPO 需要 reward_model (nn.Module, 通常是 AutoModelForSequenceClassification)
    - PPO 需要 value_model (nn.Module, 通常也是 AutoModelForSequenceClassification)
    - PPO 的数据集格式为已 tokenize 的 input_ids（而非 GRPO 的 prompt 字符串）
    - GRPO 使用 reward function (Callable)，更灵活
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        dataset=None,
        reward_model=None,
        reward_model_name: Optional[str] = None,
        value_model=None,
        value_model_name: Optional[str] = None,
        ref_model=None,
        peft_config=None,
    ):
        """
        初始化PPO训练器

        Args:
            config: 训练配置
            dataset: 训练数据集（需包含 "input_ids" 列，或包含 "prompt" 列会自动 tokenize）
            reward_model: 奖励模型 (nn.Module, AutoModelForSequenceClassification)。
                          如果未提供，将使用 reward_model_name 自动加载。
            reward_model_name: 奖励模型名称/路径，当 reward_model 未提供时使用。
                               默认使用与策略模型相同的模型名称。
            value_model: 价值模型 (nn.Module, AutoModelForSequenceClassification)。
                         如果未提供，将使用 value_model_name 自动加载。
            value_model_name: 价值模型名称/路径，当 value_model 未提供时使用。
                              默认使用与策略模型相同的模型名称。
            ref_model: 参考模型 (nn.Module)。如果未提供且不使用 PEFT，
                       将自动从策略模型创建一份副本。设为 None + peft_config
                       时使用 PEFT adapter 作为隐式参考。
            peft_config: PEFT配置（如LoraConfig），用于参数高效微调。
                         使用PEFT时无需额外的ref_model。
        """
        super().__init__(config)
        self.dataset = dataset
        self.reward_model = reward_model
        self.reward_model_name = reward_model_name
        self.value_model = value_model
        self.value_model_name = value_model_name
        self.ref_model = ref_model
        self.peft_config = peft_config

    def setup_model(self):
        """设置策略模型、奖励模型、价值模型和 tokenizer"""
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        print(f"📦 加载策略模型: {self.config.model_name}")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载策略模型 (CausalLM)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        print("✅ 策略模型加载完成")

        # 加载奖励模型 (SequenceClassification)
        # PPOTrainer 通过 model.score() 获取标量奖励，需要 num_labels=1
        if self.reward_model is None:
            rm_name = self.reward_model_name or self.config.model_name
            print(f"📦 加载奖励模型: {rm_name}")
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                rm_name,
                trust_remote_code=True,
                num_labels=1,
                ignore_mismatched_sizes=True,
            )
            print("✅ 奖励模型加载完成")

        # 加载价值模型 (SequenceClassification)
        # PPOTrainer 使用 value_model.score() 输出逐 token 的标量价值估计
        if self.value_model is None:
            vm_name = self.value_model_name or self.config.model_name
            print(f"📦 加载价值模型: {vm_name}")
            self.value_model = AutoModelForSequenceClassification.from_pretrained(
                vm_name,
                trust_remote_code=True,
                num_labels=1,
                ignore_mismatched_sizes=True,
            )
            print("✅ 价值模型加载完成")

        # 参考模型处理：使用 PEFT 时无需单独的 ref_model
        if self.ref_model is None and self.peft_config is None:
            print("📦 创建参考模型（策略模型副本）")
            import copy
            self.ref_model = copy.deepcopy(self.model)
            print("✅ 参考模型创建完成")

    def prepare_dataset(self, dataset=None):
        """将数据集转换为 PPOTrainer 所需的 tokenized 格式

        PPOTrainer 期望数据集包含 'input_ids' 列（tokenized prompt tensors）。
        如果数据集包含 'prompt' 列（字符串），将自动进行 tokenization。

        Args:
            dataset: 可选的数据集，未提供时使用 self.dataset

        Returns:
            tokenized 后的 Dataset
        """
        ds = dataset or self.dataset
        if ds is None:
            raise ValueError("数据集未设置，请提供训练数据集")

        # 检查是否已经 tokenized
        if "input_ids" in ds.column_names:
            print("✅ 数据集已包含 input_ids，跳过 tokenization")
            return ds

        if "prompt" not in ds.column_names:
            raise ValueError(
                "数据集必须包含 'input_ids' 或 'prompt' 列。"
                "请使用 create_rl_dataset() 创建数据集，或手动添加 'prompt' 列。"
            )

        if self.tokenizer is None:
            raise ValueError("tokenizer 尚未初始化，请先调用 setup_model()")

        print("📝 正在 tokenize 数据集...")
        tokenizer = self.tokenizer

        def tokenize_fn(examples):
            return tokenizer(
                examples["prompt"],
                padding="max_length",
                truncation=True,
                max_length=self.config.max_length,
                return_tensors=None,
            )

        tokenized_ds = ds.map(
            tokenize_fn,
            batched=True,
            remove_columns=[
                col for col in ds.column_names if col not in ("input_ids", "attention_mask")
            ],
        )
        tokenized_ds.set_format(type="torch")
        print(f"✅ Tokenization 完成，共 {len(tokenized_ds)} 条样本")
        return tokenized_ds

    def train(self):
        """开始PPO训练

        完整流程：
        1. 加载策略模型、奖励模型、价值模型
        2. 准备数据集（tokenize）
        3. 配置 PPOConfig
        4. 创建 PPOTrainer 并开始训练

        Returns:
            PPOTrainer 实例
        """
        import os
        from trl import PPOConfig, PPOTrainer

        # 静默实验性警告
        os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"

        if self.model is None:
            self.setup_model()

        if self.dataset is None:
            raise ValueError("数据集未设置，请提供训练数据集")

        # 准备 tokenized 数据集
        tokenized_dataset = self.prepare_dataset()

        # 确定 report_to 参数
        report_to = []
        if self.config.use_wandb:
            report_to.append("wandb")
        if self.config.use_tensorboard:
            report_to.append("tensorboard")
        if not report_to:
            report_to = ["none"]

        # 配置PPO训练参数
        training_args = PPOConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            fp16=self.config.use_fp16,
            bf16=self.config.use_bf16,
            report_to=report_to,
            # PPO 特定参数
            num_ppo_epochs=4,
            kl_coef=0.05,
            cliprange=0.2,
            vf_coef=0.1,
            cliprange_value=0.2,
            gamma=1.0,
            lam=0.95,
            whiten_rewards=False,
            # 生成参数
            response_length=self.config.max_new_tokens,
            temperature=self.config.temperature,
            stop_token="eos",
            # 需要 eval_dataset 才能使用 num_sample_generations
            num_sample_generations=0,
        )

        # 创建详细日志回调
        logging_callback = DetailedLoggingCallback(
            num_epochs=self.config.num_train_epochs
        )

        # 创建PPO训练器
        self.trainer = PPOTrainer(
            args=training_args,
            processing_class=self.tokenizer,
            model=self.model,
            ref_model=self.ref_model,
            reward_model=self.reward_model,
            value_model=self.value_model,
            train_dataset=tokenized_dataset,
            callbacks=[logging_callback],
            peft_config=self.peft_config,
        )

        print("\n🚀 开始PPO训练...")
        print(f"{'='*80}\n")
        self.trainer.train()
        print(f"\n{'='*80}")
        print("✅ PPO训练完成")

        return self.trainer

