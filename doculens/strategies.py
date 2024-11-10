import random

import pandas as pd
import rerankers


# ----- PROMPTS ENGINEER -----
def chain_of_thoughts(df: pd.DataFrame, n_samples: int = 30) -> list[str]:
    "Get a list of examplars for the training dataset"
    rand_idx = random.sample(range(0, len(df)), n_samples)
    result_df = df.iloc[rand_idx][["question", "context"]]
    questions = result_df.question.to_list()
    contexts = result_df.context.to_list()
    results = []

    for question, context in zip(questions, contexts):
        result = (
            question + ": " + context[2:-2]
        )  # Remove open/close brackets and quote ([] and '')
        results.append(result)
    return results


def query_refining(query: str) -> str: ...


# ----- RE-RANKING RESULTS -----
def reranking(query: str, contexts: list, cids: list, reranker: rerankers.Reranker):
    results = reranker.rank(query=query, documents=contexts, doc_ids=[cids])

    return results
