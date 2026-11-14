"""ModelDirector - Stateless AI model selection engine."""

from modeldirector.config import Config, Cost, ModelProfile, PolicyConfig, SelectorConfig
from modeldirector.loader import load_config
from modeldirector.models import ModelScore, SelectionResult
from modeldirector.policy import Policy, select_model
from modeldirector.selector import ModelDirector

__version__ = "0.2.0"

__all__ = [
    "Config",
    "Cost",
    "ModelProfile",
    "PolicyConfig",
    "SelectorConfig",
    "ModelScore",
    "SelectionResult",
    "Policy",
    "select_model",
    "ModelDirector",
    "load_config",
]
