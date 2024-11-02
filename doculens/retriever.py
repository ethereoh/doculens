"""
This script is for data manipulation: 
    1. Move data from dataset into milvus. 
    2. Retrieve by similarity search.
"""

import pandas as pd

from .config import MilvusDBConfig, DatasetConfig
from .db_connection import MilvusDBConnection
from .helpers import process_data_in_batches

class DoculensRetreiver:
    "Actions on Database"

    def __init__(self, 
                 embedding_model, 
                 ds_conf = DatasetConfig(), 
                 mlv_conf = MilvusDBConfig()): 

        
        self.ds_conf = ds_conf
        self.mlv_conf = mlv_conf

        # Setup Embedding model
        self.embedding_model = embedding_model

        # Setup MilvusDB Connection
        connection = MilvusDBConnection(config=mlv_conf)
        connection.create_collection()

        self.client = connection.client

        self.setup_db()

    def setup_db(self):
        "Create an instance to database"

        # 1. Convert embedding value from string to float
        vector_df = pd.read_csv(self.DS_CONFIG.vector_src_dir, index_col=0) 
        vector_df['embeddings'] = vector_df['embeddings'].apply(lambda x: self._convert_string_to_float_df(x))

        for batch in process_data_in_batches(vector_df, batch_size=1000):
            data = [batch.iloc[idx].to_dict() for idx in range(len(batch))]

            # Insert records
            res = self.client.insert(
                collection_name=self.mlv_conf.collection_name,
                data=data
            )

            print(res) 

    def retrieve(self, query: str | list[str]) -> dict:
        "Retrieve an instance"
        ...
        sentence_embedding = self.embedding_model.embed(query)
        search_params = {
            'metric_type': self.mlv_conf.metric_type, 
            'params': self.mlv_conf.params
        }
        result = self.client.search(
            collection_name=self.mlv_conf.collection_name, 
            data=sentence_embedding, 
            limit=self.mlv_conf.limit, 
            output_fields=self.mlv_conf.output_fields, 
            search_params=search_params
        )
        assert result[0] == dict
        return result[0]

    def _convert_string_to_float_df(self, sample): 
        # Remove quotes and square brackets
        string = sample[1:-1]  # Remove the first and last characters

        # Split the string into a list of strings
        float_strings = string.split()

        # Convert each string to a float
        float_list = [float(s) for s in float_strings]

        return float_list