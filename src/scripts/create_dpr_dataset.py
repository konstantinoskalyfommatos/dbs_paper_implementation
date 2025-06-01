"""Creates a similarity dictionary for DPR training from synthetic references."""

from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

model = SentenceTransformer("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True)
model.to("cuda")

df = pd.read_csv("data/new_train.csv")

sentences = df["synthetic ref"].tolist()

with torch.no_grad(): 
    embeddings = model.encode(sentences)
    similarities = model.similarity(embeddings, embeddings)

similarity_dict = {}
LEN_SENTENCES = len(sentences)
for i in tqdm(range(LEN_SENTENCES)):
    name = df.iloc[i]["Synthetic Name"]
    same_name_indices = df[df["Synthetic Name"] == name].index
    
    sorted_indices = np.argsort(similarities[i])
    hard_negative_indices = sorted_indices[:5]
    negative_indices = sorted_indices[(LEN_SENTENCES // 2 - 2) : (LEN_SENTENCES // 2 + 3)]

    similarity_dict[i] = {
        "ref": sentences[i],
        "table_str": df.iloc[i]["synthetic orig_mr"],
        "positive": {int(idx): sentences[idx] for idx in same_name_indices},
        "negative": {int(idx): sentences[idx] for idx in negative_indices},
        "hard_negative": {int(idx): sentences[idx] for idx in hard_negative_indices}
    }


with open("data/similarity_dict.json", "w") as f:
    json.dump(similarity_dict, f, indent=4)
    