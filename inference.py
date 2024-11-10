import logging

import pandas as pd

from doculens.retriever import DoculensRetreiver
from doculens.models import EmbeddingModel
from doculens.helpers import process_data_in_batches
from doculens.config import DatasetConfig


# Setup Config
ds_conf = DatasetConfig()

# Setup Retriever
print("Setting Doculens Retriever")
embedding_model = EmbeddingModel()
retriever = DoculensRetreiver(embedding_model=embedding_model)


# Setup public test
print("Reading Dataframe")
corpus_df = pd.read_csv(ds_conf.corpus_dir)
test_df = pd.read_csv(ds_conf.public_test_dir)


# Start getting results from public test
print("Start Infering: Answering legal questions")
result_dict = {"question": [], "qid": [], "context": [], "cid": []}

idx = 0
for batch in process_data_in_batches(test_df, batch_size=1000):
    print("[INFO] Inference on batch {idx}")
    # Iterate via each instance in batch
    for idx in range(len(batch)):
        question = batch.iloc[idx]["question"]
        qid = batch.iloc[idx]["qid"]

        # Retrieve relevant context
        contexts = []
        cids = []

        try:
            result = retriever.retrieve(question)
            print(result)
            for res in result[0]:
                res_entity = res["entity"]
                contexts.append(res_entity["context"])
                cids.append(res_entity["cid"])
        except:
            contexts = [None]
            cids = [-1]

        result_dict["question"].append(question)
        result_dict["qid"].append(qid)
        result_dict["context"].append(contexts)
        result_dict["cid"].append(cids)
    idx += 1

# Get result
result_df = pd.DataFrame(result_dict)
result_df.to_csv("./public_test_result.csv")
