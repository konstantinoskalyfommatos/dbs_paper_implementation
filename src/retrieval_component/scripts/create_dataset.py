"""Creates a similarity dictionary for DPR training from synthetic references."""

from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import random
random.seed(42)



def table_to_dict(table: str) -> tuple[dict, dict]:
    """Returns two dictionaries from a string representation of a table.
    
    The first dictionary is the full table, and the second is a truncated version."""
    table = table.strip()
    pairs = table.split(', ')
    dict_table = {}
    for pair in pairs:
        key_value = pair.split("[", 1)
        if len(key_value) == 2:
            key = key_value[0].strip()
            value = key_value[1].replace("]", "").strip()
            dict_table[key] = value

    # Randomly truncate
    trunc_dict_table = dict_table.copy()
    keys_without_name = [key for key in trunc_dict_table.keys() if key != 'name']

    if len(trunc_dict_table) > 1 and keys_without_name:
        key_to_empty = random.choice(keys_without_name)
        temp_keys_without_name_and_empty = [k for k in keys_without_name if k != key_to_empty]

        trunc_dict_table[key_to_empty] = ''

        if temp_keys_without_name_and_empty:
            key_to_delete = random.choice(temp_keys_without_name_and_empty)
            del trunc_dict_table[key_to_delete]

        elif (
            len(keys_without_name) == 1 and 
            'name' in trunc_dict_table and 
            len(trunc_dict_table) > 1
        ):
            pass

    return dict_table, trunc_dict_table


def serialize_dict(dict_table: dict, format_type: str = "columnar") -> str:
    """Serializes a dictionary into a string format."""
    results = []
    if format_type == "columnar":
        for key, value in dict_table.items():
            results.append(f"<r>{key}<r>{value}<r>")
        return '<c>'.join(results)
    
    elif format_type == "csv":
        columns_row = "|".join([key for key in dict_table.keys()])
        values_row = "|".join([value for value in dict_table.values()])
        return f"{columns_row}\n{values_row}"
    
    else:
        raise ValueError("Unsupported format type. Use 'columnar' or 'csv'.")


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

        full_dict_table, trunc_dict_table = table_to_dict(df.iloc[i]["synthetic orig_mr"])

        similarity_dict[i] = {
            "truncated_serialized_query": serialize_dict(trunc_dict_table, format_type="columnar"),
            "serialized_query": serialize_dict(full_dict_table, format_type="columnar"),
            "truncated_serialized_query_csv": serialize_dict(trunc_dict_table, format_type="csv"),
            "serialized_query_csv": serialize_dict(full_dict_table, format_type="csv"),
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
