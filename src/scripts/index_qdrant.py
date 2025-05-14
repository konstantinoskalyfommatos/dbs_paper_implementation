from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import pandas as pd
import numpy as np


client = QdrantClient(
   host="qdrant", 
   port=6333,
)

COLLECTION_NAME = "e2e_documents"

if not client.collection_exists(COLLECTION_NAME):
   client.create_collection(
      collection_name=COLLECTION_NAME,
      vectors_config=VectorParams(size=768, distance=Distance.COSINE),
   )


model = "my_model"
df = pd.read_csv("/home/retrieval_component/data/new_train.csv")
df["vector"] = None
df["vector"] = df["synthetic ref"].apply(
   lambda x: np.random.rand(768)
)

for idx, row in df.iterrows():
   print(idx)
   client.upsert(
      collection_name=COLLECTION_NAME,
      points=[
         PointStruct(
            id=idx,
            vector=row["vector"].tolist(),
            payload={
               key: value for key, value in row.to_dict().items()
               if key != "vector"
            }
         )
      ]
   )

print("Finished")