import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any
import json
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(project_root, "src"))

from augmentation_component.finetune_bart_1 import CustomBart, collate_fn


class BartDataset(Dataset):
    def __init__(self, data: dict, tokenizer, max_length=256):
        self.data_dict = data
        self.data_keys = sorted(data.keys(), key=int)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_keys)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item_key = self.data_keys[idx]
        item = self.data_dict[item_key]
        
        serialized_query = item["truncated_serialized_query"]
        retrieved_documents = item["ground_truth_retrieved"]
        retrieved_documents.append(item["positive"])

        query_enc = self.tokenizer(
            serialized_query, 
            truncation=True, 
            padding="max_length",
            max_length=self.max_length, 
            return_tensors="pt"
        )

        retr_enc = [
            self.tokenizer(
                doc,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            ) for doc in retrieved_documents
        ]
        
        labels = self.tokenizer(
            item["serialized_query"], 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_length, 
            return_tensors="pt"
        )

        return {
            "query_input_ids": query_enc["input_ids"].squeeze(0),
            "query_attention_mask": query_enc["attention_mask"].squeeze(0),
            "retr_input_ids_list": [enc["input_ids"].squeeze(0) for enc in retr_enc],
            "retr_attention_mask_list": [enc["attention_mask"].squeeze(0) for enc in retr_enc],
            "labels": labels["input_ids"].squeeze(0)
        }

def main():

    device = "cuda"
    model = CustomBart.from_pretrained('./models/bart_finetuned')
    model.to(device)
    model.eval()

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    new_tokens = ["<r>", "<c>"]
    tokenizer.add_tokens(new_tokens)


    try:
        with open("data/dataset_dict.json", "r") as f:
            train_dataset_dict = json.load(f)
    except FileNotFoundError:
        print("Error: data/dataset_dict.json not found. Please ensure the path is correct.")
        exit()

    dataset = BartDataset(train_dataset_dict, tokenizer, max_length=256)

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn
    )

    for batch in dataloader:

        query_input_ids = batch['query_input_ids'].to(device)
        query_attention_mask = batch['query_attention_mask'].to(device)
        retr_input_ids_list = [ids.to(device) for ids in batch['retr_input_ids_list']]
        retr_attention_mask_list = [mask.to(device) for mask in batch['retr_attention_mask_list']]
        
        outputs = model.generate(
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            retr_input_ids_list=retr_input_ids_list,
            retr_attention_mask_list=retr_attention_mask_list
        )

        print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
