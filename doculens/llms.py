import ollama

from .config import LanguageConfig


class OllamaAgent:
    def __init__(self, config: LanguageConfig):
        self.config = config
        self.model_name = config.model_name

    def _init_model(self): ...

    def invoke(self, input: str) -> str | list[str]: ...

    def _check_model(self, model_name) -> bool:
        "Check if model exists"
        ...
