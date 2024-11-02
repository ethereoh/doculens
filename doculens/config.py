from dataclasses import dataclass, field

import torch


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    

@dataclass
class ModelConfig(Config):
    model_name: str = "keepitreal/vietnamese-sbert"


@dataclass
class DatasetConfig: 
    data_root : str = './db'
    corpus_dir: str =  f"{data_root}/corpus.csv"
    train_dir: str = f"{data_root}/train.csv" 
    vector_src_dir: str = f"{data_root}/vector_db_src.csv"
    public_test_dir: str = f"{data_root}/public_test.csv"

@dataclass
class MilvusDBConfig:
    data_root : str = './db'

    db_name: str = f"bkai_milvus.db" # Change this to place the db to where you want
    collection_name: str = "bkai_vectordb"
    limit: int = 20  # This is top_k results
    output_fields : list = field(default_factory=lambda : ['question', 'context', 'cid'])
    metric_type: str = "L2" # Possible values are IP, L2, COSINE, JACCARD, and HAMMING


    # More details at: https://milvus.io/api-reference/pymilvus/v2.4.x/MilvusClient/Vector/search.md
    params : dict = field(default_factory=lambda :  {})

    # Dataset config: this part requires dataset's EDA. 
    dimension: int = 768 # Length of the embedding vector (1, embed_len)
    primary_field_name: str ="id"
    id_type:str ="int"
    vector_field_name: str ="embeddings"
    auto_id: bool = False