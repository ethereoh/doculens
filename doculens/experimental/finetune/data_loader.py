from pathlib import Path

import datasets
from unsloth.chat_templates import get_chat_template, standardize_sharegpt

from .preprocessing import convert_data_to_FineTome_100k_format


def load_dataset(
    train_file: str | Path,
    tokenizer: any,
    progress: bool = False,
) -> datasets.Dataset:
    """
    Load dataset from csv file
    """
    assert Path(train_file).suffix == ".csv", "Only csv files are supported"
    conversations_list = convert_data_to_FineTome_100k_format(
        train_file, progess=progress
    )

    dataset = datasets.Dataset.from_dict({"conversations": conversations_list})

    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {
            "text": texts,
        }

    dataset = standardize_sharegpt(dataset)
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
    )

    return dataset
