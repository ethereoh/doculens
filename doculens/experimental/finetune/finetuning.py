from transformers import DataCollatorForSeq2Seq, TrainingArguments
from trl import SFTTrainer
from unsloth import is_bfloat16_supported

from .config import FinetuneConfig
from .data_loader import load_dataset
from .models import Model

model_config = Model()
ft_config = FinetuneConfig()

dataset = load_dataset(
    train_file="./dataset/preprocessed.csv", tokenizer=model_config.tokenizer
)
output_dir = "./models"
trainer = SFTTrainer(
    model=model_config.model,
    tokenizer=model_config.tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=ft_config.max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=model_config.tokenizer),
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_dir,
        report_to="none",
    ),
)
