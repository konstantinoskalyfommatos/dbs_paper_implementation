"""Creates a similarity dictionary for DPR training from synthetic references."""

from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import random
random.seed(42)

def serialize_query(table: str, truncate: bool = True) -> str:
    results = []
    table = table.strip()
    pairs = table.split(', ')
    dict_table = {}
    for pair in pairs:
        key_value = pair.split("[", 1)
        if len(key_value) == 2:
            key = key_value[0].strip()
            value = key_value[1].replace("]", "").strip()
            dict_table[key] = value

    if truncate:
        keys_without_name = [key for key in dict_table.keys() if key != 'name']

        if len(dict_table) > 1 and keys_without_name:
            key_to_empty = random.choice(keys_without_name)
            # Ensure the key_to_empty is not the only non-name key if we intend to delete another one
            temp_keys_without_name_and_empty = [k for k in keys_without_name if k != key_to_empty]

            dict_table[key_to_empty] = ''  # Modify a random non-'name' field to be empty

            # Delete another random non-'name' field if available
            if temp_keys_without_name_and_empty:  # If there are other non-name keys left
                key_to_delete = random.choice(temp_keys_without_name_and_empty)
                del dict_table[key_to_delete]
            # If after emptying, only 'name' and the (now empty) key_to_empty remains,
            # or if there was only one non-name key to begin with, we can't delete another distinct non-name key.
            elif len(keys_without_name) == 1 and 'name' in dict_table and len(
                    dict_table) > 1:  # Only one non-name key existed
                pass  # Cannot delete another non-name key as it was the one emptied

    for key, value in dict_table.items():
        results.append(f"<r>{key}<r>{value}<r>")
    return '<c>'.join(results)


def main():
    # NOTE: Run from the root directory of the project
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
            "truncated_serialized_query": serialize_query(df.iloc[i]["synthetic orig_mr"], truncate=True),
            "serialized_query": serialize_query(df.iloc[i]["synthetic orig_mr"], truncate=False),
            "ground_truth_retrieved": df[df["Synthetic Name"]== name]['synthetic ref'].to_list(),
            "table_str": df.iloc[i]["synthetic orig_mr"],
            "positive": sentences[i],
            "negative": {int(idx): sentences[idx] for idx in negative_indices},
            "hard_negative": {int(idx): sentences[idx] for idx in hard_negative_indices}
        }

    with open("data/dataset_dict.json", "w") as f:
        json.dump(similarity_dict, f, indent=4)
    

if __name__ == "__main__":
    main()
