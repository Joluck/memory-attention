
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.memory.configuration_memory import MemoryConfig
from fla.models.memory.modeling_memory import MemoryForCausalLM, MemoryModel

AutoConfig.register(MemoryConfig.model_type, MemoryConfig, exist_ok=True)
AutoModel.register(MemoryConfig, MemoryModel, exist_ok=True)
AutoModelForCausalLM.register(MemoryConfig, MemoryForCausalLM, exist_ok=True)


__all__ = ['MemoryConfig', 'MemoryForCausalLM', 'MemoryModel']
