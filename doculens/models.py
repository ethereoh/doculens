from unsloth import FastLanguageModel

from .config import FinetuneConfig


class Model(object):
    def __init__(self, config=FinetuneConfig(), auto: bool = True):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,  # or choose "unsloth/Llama-3.2-1B-Instruct"
            max_seq_length=config.max_seq_length,
            dtype=config.dtype,
            load_in_4bit=config.load_in_4bit,
        )

        if auto:
            self.model = self._quantize_model(self.model)

    def _quantize_model(self, model):
        return FastLanguageModel.get_peft_model(
            model,
            r=32,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=32,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=3407,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )
