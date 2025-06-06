import torch
from transformers import DPRQuestionEncoder, DPRContextEncoder
from transformers import DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
import torch
from torch.nn import functional as F
from torch import nn
import random
from typing import Dict
import json
from sklearn.metrics.pairwise import cosine_similarity

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
        results.append(f"<R>{key}<R>{value}<R>")
    return '<C>'.join(results)





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
        cos_sim = cosine_similarity(q_embeds.cpu(), p_embeds.cpu())
        target_labels = torch.arange(q_embeds.size(0), device=scores.device)
        loss = F.cross_entropy(scores, target_labels)

        return {"loss": loss, "scores": scores, "q_embeds": q_embeds, "p_embeds": p_embeds, 'cosine_similarity': cos_sim}




def prepare_dataset(data, idx, question_tokenizer, context_tokenizer, max_length=256) -> Dict[str, torch.Tensor]:

    item = data[str(idx)]
    serialized_query = _serialize_query(item["table_str"])

    positive = item["positive"]
    # serialized_query = 'The Roasted Retreat Indian restaurant offers high-end dining  in the riverside area.'

    query_enc = question_tokenizer(serialized_query, truncation=True, padding="max_length",
                                 max_length=max_length, return_tensors="pt")
    pos_enc = context_tokenizer(positive, truncation=True, padding="max_length", max_length=max_length,
                                 return_tensors="pt")

    result = {
        "query_input_ids": query_enc["input_ids"].to(device),
        "query_attention_mask": query_enc["attention_mask"].to(device),
        "pos_input_ids": pos_enc["input_ids"].to(device),
        "pos_attention_mask": pos_enc["attention_mask"].to(device),
    }
        # Move tensors to the same device as the model
    return result

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

    question_encoder_name = "facebook/dpr-question_encoder-single-nq-base"
    context_encoder_name = "facebook/dpr-ctx_encoder-single-nq-base"


    # question_encoder = DPRQuestionEncoder.from_pretrained(question_encoder_name)
    # ctx_encoder = DPRContextEncoder.from_pretrained(context_encoder_name)

    question_encoder = DPRQuestionEncoder.from_pretrained('dpr_finetuned_question_encoder')
    ctx_encoder = DPRContextEncoder.from_pretrained('dpr_finetuned_ctx_encoder')




    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(question_encoder_name)
    new_tokens = ["<R>", "<C>"]
    tokens_to_add = [token for token in new_tokens if token not in q_tokenizer.vocab]
    if tokens_to_add:
        q_tokenizer.add_tokens(tokens_to_add)
        question_encoder.resize_token_embeddings(len(q_tokenizer))

    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(context_encoder_name)

    # Reconstruct your DPRDualEncoder
    model = DPRDualEncoder(question_encoder, ctx_encoder)
    model.eval()  # Set to evaluation mode

    # If you need to use GPU
    device = 'cuda'
    model.to(device)

    with open('data/similarity_dict.json', 'r') as f:
        data = json.load(f)

    result = prepare_dataset(data, 1, question_tokenizer=q_tokenizer, context_tokenizer=ctx_tokenizer)

    with torch.no_grad():
        result = prepare_dataset(data, 2, question_tokenizer=q_tokenizer, context_tokenizer=ctx_tokenizer) 
        outputs = model(**result)
        print(outputs['scores'], outputs['cosine_similarity'], outputs['loss'])
