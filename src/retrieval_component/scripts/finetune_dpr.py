from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch import nn
import random
import json
# Removed DataCollatorWithPadding as we are making a more specific custom one
# from transformers import DataCollatorWithPadding
from typing import Any, Dict, List, Union, Optional


class DPRDataset(Dataset):
    def __init__(self, data: Dict[str, Dict[str, Any]], question_tokenizer, context_tokenizer, max_length=256):
        self.data_dict = data  # Keep original dict
        self.data_keys = sorted(data.keys(), key=int)  # Assumes keys are "0", "1", ...
        self.q_tokenizer = question_tokenizer
        self.ctx_tokenizer = context_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_keys)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item_key = self.data_keys[idx]
        item = self.data_dict[item_key]
        serialized_query = self.data_dict[item_key]["truncated_serialized_query"]

        positive = item["positive"]

        query_enc = self.q_tokenizer(serialized_query, truncation=True, padding="max_length",
                                     max_length=self.max_length, return_tensors="pt")
        pos_enc = self.ctx_tokenizer(positive, truncation=True, padding="max_length", max_length=self.max_length,
                                     return_tensors="pt")

        result = {
            "query_input_ids": query_enc["input_ids"].squeeze(0),
            "query_attention_mask": query_enc["attention_mask"].squeeze(0),
            "pos_input_ids": pos_enc["input_ids"].squeeze(0),
            "pos_attention_mask": pos_enc["attention_mask"].squeeze(0),
        }
        return result

    @staticmethod
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


class DPRDualEncoder(nn.Module):
    def __init__(self, question_encoder, ctx_encoder):
        super().__init__()
        self.question_encoder = question_encoder
        self.ctx_encoder = ctx_encoder

    def forward(self,
                query_input_ids,
                query_attention_mask,
                pos_input_ids,
                pos_attention_mask,
                labels=None,
                **kwargs):
        q_outputs = self.question_encoder(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask
        )
        q_embeds = q_outputs.pooler_output

        p_outputs = self.ctx_encoder(
            input_ids=pos_input_ids,
            attention_mask=pos_attention_mask
        )
        p_embeds = p_outputs.pooler_output
        scores = torch.matmul(q_embeds, p_embeds.T)
        target_labels = torch.arange(q_embeds.size(0), device=scores.device)
        loss = F.cross_entropy(scores, target_labels)

        return {"loss": loss, "scores": scores, "q_embeds": q_embeds, "p_embeds": p_embeds}


class DPRTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


class CustomDPRDataCollator:
    """
    Data collator that specifically handles batches of DPR data with prefixed keys.
    It uses the tokenizer's pad method for each logical group of inputs (e.g., query, positive).
    Assumes that DPRDataset provides items where input_ids and attention_masks are 1D tensors
    already padded to a max_length. This collator will then convert them to lists of ints
    to be re-padded/batched by tokenizer.pad into final batch tensors.
    """

    def __init__(self, tokenizer, padding: Union[bool, str] = True, max_length: Optional[int] = None,
                 pad_to_multiple_of: Optional[int] = None, return_tensors: str = "pt"):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not features:
            return {}

        batch = {}

        # Identify unique prefixes (e.g., "query", "pos") from the keys
        prefixes = set()
        for key in features[0].keys():
            if "_input_ids" in key:
                prefixes.add(key.split("_input_ids")[0])

        for prefix in prefixes:
            input_ids_key = f"{prefix}_input_ids"
            attention_mask_key = f"{prefix}_attention_mask"
            token_type_ids_key = f"{prefix}_token_type_ids"  # For completeness, though DPR might not use them

            if input_ids_key not in features[0]:
                continue  # Skip if this group doesn't have input_ids

            # Convert 1D tensors from dataset back to list of lists of ints for tokenizer.pad
            input_ids_list = [f[input_ids_key].tolist() for f in features]

            # Prepare dictionary for padding this specific group
            to_pad_group = {"input_ids": input_ids_list}

            if attention_mask_key in features[0]:
                attention_mask_list = [f[attention_mask_key].tolist() for f in features]
                to_pad_group["attention_mask"] = attention_mask_list

            if token_type_ids_key in features[0] and hasattr(self.tokenizer,
                                                             'model_input_names') and "token_type_ids" in self.tokenizer.model_input_names:
                token_type_ids_list = [f[token_type_ids_key].tolist() for f in features]
                to_pad_group["token_type_ids"] = token_type_ids_list

            # Use the tokenizer's pad method for the current group
            padded_group = self.tokenizer.pad(
                to_pad_group,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )

            # Add padded results to the final batch with original prefixed keys
            batch[input_ids_key] = padded_group["input_ids"]
            if "attention_mask" in padded_group and attention_mask_key in features[0]:
                batch[attention_mask_key] = padded_group["attention_mask"]
            if "token_type_ids" in padded_group and token_type_ids_key in features[
                0] and "token_type_ids" in padded_group:  # Check if tokenizer produced it
                batch[token_type_ids_key] = padded_group["token_type_ids"]

        # Handle any other keys not part of these input groups (e.g., metadata)
        all_processed_keys = set(batch.keys())
        for key in features[0].keys():
            if key not in all_processed_keys:
                values = [f[key] for f in features]
                if isinstance(values[0], torch.Tensor):
                    batch[key] = torch.stack(values)
                elif self.return_tensors == "pt":
                    try:
                        batch[key] = torch.tensor(values)
                    except Exception:
                        batch[key] = values  # Keep as list if tensor conversion fails
                else:
                    batch[key] = values

        return batch


if __name__ == "__main__":
    question_encoder_name = "facebook/dpr-question_encoder-single-nq-base"
    context_encoder_name = "facebook/dpr-ctx_encoder-single-nq-base"

    question_encoder = DPRQuestionEncoder.from_pretrained(question_encoder_name)
    ctx_encoder = DPRContextEncoder.from_pretrained(context_encoder_name)

    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(question_encoder_name)
    new_tokens = ["<r>", "<c>"]
    tokens_to_add = [token for token in new_tokens if token not in q_tokenizer.vocab]
    if tokens_to_add:
        q_tokenizer.add_tokens(tokens_to_add)
        question_encoder.resize_token_embeddings(len(q_tokenizer))

    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(context_encoder_name)

    try:
        with open("data/dataset_dict.json", "r") as f:
            train_dataset_dict = json.load(f)
    except FileNotFoundError:
        print("Error: data/dataset_dict.json not found. Please ensure the path is correct.")
        exit()

    train_dataset = DPRDataset(train_dataset_dict, q_tokenizer, ctx_tokenizer, max_length=256)  # max_length passed here

    training_args = TrainingArguments(
        output_dir="./dpr_finetuned",
        per_device_train_batch_size=32,
        learning_rate=1e-5,
        num_train_epochs=1,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="epoch",
        report_to="none"
    )


    # Use the new CustomDPRDataCollator
    data_collator = CustomDPRDataCollator(
        tokenizer=q_tokenizer,
        padding=True,  # This will pad to the longest in the batch for each group
        max_length=256,  # DPRDataset already pads to 256, this acts as a cap / re-assurance
        return_tensors="pt"
    )

    model = DPRDualEncoder(question_encoder, ctx_encoder)

    trainer = DPRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=q_tokenizer,
        data_collator=data_collator  # Use the custom collator
    )



    print("Starting training...")
    trainer.train()
    model.question_encoder.save_pretrained('./dpr_finetuned_question_encoder')
    model.ctx_encoder.save_pretrained('./dpr_finetuned_ctx_encoder')
    print("Training finished.")