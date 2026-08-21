"""
Capa de abstracción para modelos de lenguaje (LLM).
"""

from .base import BaseLLMClient, RuleBasedLLMClient

__all__ = ["BaseLLMClient", "RuleBasedLLMClient"]
