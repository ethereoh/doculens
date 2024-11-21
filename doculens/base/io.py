from pydantic import BaseModel


class DocuRequest(BaseModel):
    question: str
    qid: str
    rerank: bool


class DocuResponse(BaseModel):
    result: str
    text: list[str]
    cids: list[int]
