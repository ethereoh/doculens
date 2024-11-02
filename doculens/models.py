import torch
from transformers import AutoTokenizer, AutoModel

from .config import ModelConfig


class EmbeddingModel:

    def __init__(self):

        self.config = ModelConfig()
        self.model, self.tokenizer = self._init_model(
            model_name=self.config.model_name, device=self.config.device
        )

    def _init_model(self, model_name: str, device: str) -> tuple:
        "Initialize model"
        model = AutoModel.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return (model, tokenizer)

    # Mean Pooling - Take attention mask into account for correct averaging
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    # TODO: make hyperparameters read from a declarative file.
    def embed(self, payload: str | list[str]):
        tokenized_input = self.tokenizer(
            payload, padding=True, truncation=True, return_tensors="pt"
        ).to(self.config.device)
        with torch.no_grad():
            model_output = self.model(**tokenized_input)

        embedding_outputs = (
            self._mean_pooling(model_output, tokenized_input["attention_mask"])
            .detach()
            .cpu()
            .numpy()
        )

        return embedding_outputs
