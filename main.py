import rerankers
from fastapi import FastAPI, Response

from doculens.base.io import DocuRequest, DocuResponse
from doculens.config import DatasetConfig, EmbeddingConfig, RerankerConfig
from doculens.embedding import EmbeddingModel
from doculens.retriever import DoculensRetreiver

ds_conf = DatasetConfig()
rerank_conf = RerankerConfig()

embedding_model = EmbeddingModel(config=EmbeddingConfig())
retriever = DoculensRetreiver()
ranker = rerankers.Reranker(rerank_conf.model_name, lang=rerank_conf.lang)

app = FastAPI()


@app.get("/")
def hello_world():
    return Response(content="hello world")


@app.post("/inference")
def inference(payload: DocuRequest) -> DocuResponse:
    try:
        result = ""
        texts, cids = [], []

        question = payload.question
        is_rerank = payload.rerank

        search_result = retriever.retrieve(question)

        result += f"{str(payload.qid)} "
        for res in search_result[0]:
            res_entity = res["entity"]
            texts.append(res_entity["text"])
            cids.append(res_entity["cid"])

        if is_rerank:
            texts, cids = [], []
            reranked_result = ranker.rank(query=question, docs=texts, doc_ids=cids)
            for r in reranked_result.results:
                texts.append(r.text)
                cids.append(r.doc_id)
                result += f"{r.doc_id} "
        else:
            for res in search_result[0]:
                res_entity = res["entity"]
                result += f"{res_entity["cid"]} "

    except Exception as e:
        return e
