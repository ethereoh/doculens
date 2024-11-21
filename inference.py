import logging

import pandas as pd
import rerankers

from doculens.config import DatasetConfig, EmbeddingConfig
from doculens.embedding import EmbeddingModel
from doculens.helpers import process_data_in_batches
from doculens.retriever import DoculensRetreiver
from doculens.strategies import reranking

ranker = rerankers.Reranker("cross-encoder", lang="vi")

# Setup Config
ds_conf = DatasetConfig()

# Setup Retriever
print("Setting Doculens Retriever")
embedding_model = EmbeddingModel(config=EmbeddingConfig())
retriever = DoculensRetreiver()


# Setup public test
print("Reading Dataframe")
test_df = pd.read_csv(ds_conf.public_test_dir)


# Start getting results from public test
print("Start Infering: Answering legal questions")
# result_dict = {"question": [], "qid": [], "context": [], "cid": []}

idx = 0
result = ""
texts, cids = [], []
output_dir = "predict.txt"

for batch in process_data_in_batches(test_df, batch_size=1000):
    print("[INFO] Inference on batch {idx}")
    # Iterate via each instance in batch
    for idx in range(len(batch)):
        question = batch.iloc[idx]["question"]
        qid = batch.iloc[idx]["qid"]

        # Retrieve relevant context
        contexts = []
        cids = []
        result += f"{str(qid)} "

        search_result = retriever.retrieve(question)

        for res in search_result[0]:
            res_entity = res["entity"]
            texts.append(res_entity["text"])
            cids.append(res_entity["cid"])

        reranked_result = reranking(query=question, docs=texts, doc_ids=cids)

        for r in reranked_result.results:
            result += f"{r.doc_id} "

        with open(output_dir, "a+") as writer:
            print(f"{idx} question: {result}")
            result += "\n"
            writer.writelines(result)
            print("======" * 10)
        result = ""
        texts, cids = [], []
    idx += 1


# Get result
# result_df = pd.DataFrame(result_dict)
# result_df.to_csv("./public_test_result.csv")
