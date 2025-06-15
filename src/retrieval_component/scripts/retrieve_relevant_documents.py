from transformers import DPRQuestionEncoder, DPRContextEncoder
from transformers import DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
import torch
import random
from typing import Dict
import json
from qdrant_client import QdrantClient


def _serialize_query(table: str) -> str:
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



def prepare_dataset(item: dict, question_tokenizer, max_length=256) -> Dict[str, torch.Tensor]:

    serialized_query = _serialize_query(item["table_str"])

    query_enc = question_tokenizer(serialized_query, truncation=True, padding="max_length",
                                   max_length=max_length, return_tensors="pt")

    result = {
        "input_ids": query_enc["input_ids"].to(device),
        "attention_mask": query_enc["attention_mask"].to(device),
    }
    return result


if __name__ == "__main__":
    client = QdrantClient(
        host="localhost",
        port=6333,
    )

    COLLECTION_NAME = "e2e_documents_finetuned_dpr"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    question_encoder_name = "facebook/dpr-question_encoder-single-nq-base"
    context_encoder_name = "facebook/dpr-ctx_encoder-single-nq-base"

    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('./models/dpr_finetuned_question_tokenizer')
    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(context_encoder_name)


    question_encoder = DPRQuestionEncoder.from_pretrained('./models/dpr_finetuned_question_encoder')
    ctx_encoder = DPRContextEncoder.from_pretrained('./models/dpr_finetuned_ctx_encoder')

    question_encoder.eval()  # Set to evaluation mode

    # If you need to use GPU
    device = 'cuda'
    question_encoder.to(device)

    with open('data/dataset_dict_test.json', 'r') as f:
        data = json.load(f)

    with torch.no_grad():
        for rec in data.values():
            result = prepare_dataset(rec, question_tokenizer=q_tokenizer)
            outputs = question_encoder(**result)

            hits = client.query_points(
                collection_name=COLLECTION_NAME,
                query=outputs.pooler_output[0].tolist(),
                limit=5  # Return 5 closest points
            )
            rec['dpr_retrieved'] = [hits.points[i].payload['positive'] for i in range(len(hits.points))]

    with open('data/dataset_dict_test_e2e.json', 'w') as f:
        json.dump(data, f, indent=4)
