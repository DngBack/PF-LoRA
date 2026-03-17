"""
LoRA model factory for PF-LoRA experiments.

Provides a unified interface for creating PEFT LoRA models from config,
handling both Llama and Mistral architectures.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# Known target module sets per architecture
_TARGET_MODULES = {
    "attention_only": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "attention_and_mlp": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    # Mistral uses same projection names as Llama
    "mistral_attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def load_base_model(
    model_cfg: DictConfig,
    device_map: str = "auto",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> tuple:
    """
    Load base model and tokenizer from config.

    Args:
        model_cfg: Model config from configs/model/*.yaml.
        device_map: HuggingFace device_map strategy.
        load_in_4bit: Enable 4-bit quantization via bitsandbytes.
        load_in_8bit: Enable 8-bit quantization via bitsandbytes.

    Returns:
        (model, tokenizer) tuple.
    """
    model_name = model_cfg.model_name_or_path
    torch_dtype = getattr(torch, model_cfg.get("torch_dtype", "bfloat16"))
    attn_impl = model_cfg.get("attn_implementation", "eager")

    bnb_config = None
    if load_in_4bit or model_cfg.get("load_in_4bit", False):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
    elif load_in_8bit or model_cfg.get("load_in_8bit", False):
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_impl,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.padding_side != "right":
        tokenizer.padding_side = "right"

    return model, tokenizer


def build_lora_config(
    train_cfg: DictConfig,
    model_cfg: Optional[DictConfig] = None,
) -> LoraConfig:
    """
    Build a PEFT LoraConfig from training config.

    Args:
        train_cfg: Training config (configs/train/*.yaml).
        model_cfg: Optional model config to validate target modules.

    Returns:
        LoraConfig instance.
    """
    # Resolve target modules
    target_modules = list(train_cfg.get("target_modules", _TARGET_MODULES["attention_only"]))

    lora_config = LoraConfig(
        r=train_cfg.get("lora_rank", 16),
        lora_alpha=train_cfg.get("lora_alpha", 32),
        lora_dropout=train_cfg.get("lora_dropout", 0.05),
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_rslora=train_cfg.get("use_rslora", False),
    )
    return lora_config


def get_lora_model(
    model_cfg: DictConfig,
    train_cfg: DictConfig,
    device_map: str = "auto",
) -> tuple:
    """
    Load base model and wrap with LoRA adapter.

    Args:
        model_cfg: Model config.
        train_cfg: Training config with LoRA hyperparameters.
        device_map: Device mapping strategy.

    Returns:
        (peft_model, tokenizer) tuple.
    """
    model, tokenizer = load_base_model(model_cfg, device_map=device_map)

    if train_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_cfg = build_lora_config(train_cfg, model_cfg)
    peft_model = get_peft_model(model, lora_cfg)
    peft_model.print_trainable_parameters()

    return peft_model, tokenizer


def get_trainable_param_count(model) -> int:
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_adapter_modules(model, module_name_filter: Optional[str] = None) -> dict:
    """
    Return a dict of {layer_name: (A_matrix, B_matrix)} for all LoRA adapters.

    Useful for computing the protected penalty at training time.
    """
    adapters: dict = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            if module_name_filter and module_name_filter not in name:
                continue
            a = module.lora_A.get("default", None)
            b = module.lora_B.get("default", None)
            if a is not None and b is not None:
                adapters[name] = (a.weight, b.weight)
    return adapters
