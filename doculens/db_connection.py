"Connet to MilvusDB"

import pymilvus
from pymilvus import MilvusClient

class MilvusDBConnection:
    "Establish Connection to Milvus"

    def __init__(self, config):
        self.config = config
        self.connect()


    def connect(self):
        try:
            self.client = MilvusClient(self.config.db_name)
        except Exception as e:
            raise pymilvus.exceptions.ConnectError() from e
        

    def check_collection(self) -> bool: 
        return self.client.has_collection(collection_name=self.config.collection_name)

    def create_collection(self): 
        try: 
            if self.check_collection():
                pass
            else: 
                self.client.create_collection(
                    collection_name=self.config.collection_name,
                    dimension=self.config.dimension,
                    primary_field_name=self.config.primary_field_name,
                    id_type=self.config.id_type,
                    vector_field_name=self.config.vector_field_name,
                    auto_id=self.config.auto_id,                
                    metric_type=self.config.metric_type
                    )
        except Exception as e: 
            raise pymilvus.exceptions.CollectionNotExistException() from e


    def drop_collection(self):
        try: 
            if self.check_collection():
                self.client.drop_collection(collection_name=self.config.collection_name)
        except Exception as e: 
            raise pymilvus.exceptions.CollectionNotExistException() from e