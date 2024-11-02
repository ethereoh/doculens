from pathlib import Path

import pandas as pd
from tqdm import tqdm
from unsloth.chat_templates import get_chat_template


def preprocess_qa_data(row: pd.Series) -> list[dict]:
    """
    Preprocess row data from BKAI: Merging question and context.
    """
    result: list[dict] = []
    question = row["question"].replace('"', "").replace("'", "").strip("\n").strip()
    qid = row["qid"]

    cid = [int(x) for x in row["cid"].strip("[] ").split(" ") if x]
    context = row["context"].strip("[]").split("'\n")

    assert len(cid) == len(context), f"{len(cid)} != {len(context)} for {row}"
    for i in range(len(cid)):
        result.append(
            {
                "question": question,
                "context": context[i]
                .replace('"', "")
                .replace("'", "")
                .strip("\n")
                .strip(),
                "cid": cid[i],
                "qid": qid,
            }
        )

    return result


def preprocess_convert_to_finetome100k(row: pd.Series) -> list[dict]:
    """
    Preprocess row data: convert BKAI to FineTome 100k format
    """
    result: list[tuple[dict, dict]] = []
    question = row["question"]

    context = row["context"]

    result.append(
        [
            {
                "from": "human",
                "value": question,
            },
            {
                "from": "gpt",
                "value": context,
            },
        ]
    )

    return result


def convert_data_to_FineTome_100k_format(
    train_file: str | Path, progess: bool = True
) -> list[dict]:
    """
    Convert BKAI data to FineTome 100k format
    """
    assert Path(train_file).suffix == ".csv", "Only csv files are supported"

    conversations_list: list[dict] = []
    df = pd.read_csv(train_file)

    iterrows = tqdm(df.iterrows(), total=len(df)) if progess else df.iterrows()
    for _, row in iterrows:
        conversations_list.extend(preprocess_convert_to_finetome100k(row))

    return conversations_list
