from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from typing import Optional, List, Dict, Any
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
from torch import nn
import json
from transformers import BartConfig
from torch.utils.data import DataLoader
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.bart.modeling_bart import shift_tokens_right
import torch.nn.functional as F
from tqdm import tqdm


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

class CustomBart(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        # TODO: Check about resizing token embeddings
        self.proj_query = nn.Linear(config.d_model, config.d_model // 2)
        self.proj_retr = nn.Linear(config.d_model, config.d_model // 2)

    def forward(
        self,
        query_input_ids: torch.LongTensor,
        query_attention_mask: torch.Tensor,
        retr_input_ids_list: List[torch.LongTensor],
        retr_attention_mask_list: List[torch.Tensor],
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Any:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # Encode query and documents
        query_outputs = self.model.encoder(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_encoder_hidden_states = query_outputs[0]

        retr_outputs_list = [
            self.model.encoder(
                input_ids=retr_input_ids,
                attention_mask=retr_attention_mask,
                return_dict=return_dict,
            )[0]
            for retr_input_ids, retr_attention_mask in zip(retr_input_ids_list, retr_attention_mask_list)
        ]
        
        # Project encoder outputs
        proj_query = self.proj_query(query_encoder_hidden_states)
        proj_retr_list = [self.proj_retr(retr_output) for retr_output in retr_outputs_list]

        # Calculate interpolation weights
        weights = []
        for i, proj_doc_i in enumerate(proj_retr_list):
            # Dot product with query
            query_doc_similarity = torch.einsum('bsh,bsh->bs', proj_query, proj_doc_i).sum(dim=1)
            
            # Dot product with other documents
            doc_doc_similarity = 0
            for j, proj_doc_j in enumerate(proj_retr_list):
                if i != j:
                    doc_doc_similarity += torch.einsum('bsh,bsh->bs', proj_doc_i, proj_doc_j).sum(dim=1)
            
            weights.append(query_doc_similarity + doc_doc_similarity)

        weights_tensor = torch.stack(weights, dim=1)
        softmax_weights = F.softmax(weights_tensor, dim=1)

        # Compute weighted average of document encoder outputs
        weighted_doc_outputs = torch.zeros_like(retr_outputs_list[0])
        for i, retr_output in enumerate(retr_outputs_list):
            weighted_doc_outputs += retr_output * softmax_weights[:, i].unsqueeze(-1).unsqueeze(-1)
            
        # Concatenate query and weighted documents
        combined_encoder_outputs = torch.cat((query_encoder_hidden_states, weighted_doc_outputs), dim=1)
        
        # Create attention mask for the combined output
        # Assuming all retrieved documents have the same length as the query after padding
        combined_attention_mask = torch.cat((query_attention_mask, retr_attention_mask_list[0]), dim=1)


        # Decoder
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=combined_encoder_outputs,
            encoder_attention_mask=combined_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(decoder_outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=combined_encoder_outputs,
        )

# def collate_fn(batch):
#     query_input_ids = torch.stack([item['query_input_ids'] for item in batch])
#     query_attention_mask = torch.stack([item['query_attention_mask'] for item in batch])
#     labels = torch.stack([item['labels'] for item in batch])
    
#     max_retr = max(len(item['retr_input_ids_list']) for item in batch)

#     # Pad retrieved documents to the max number in the batch
#     padded_retr_input_ids = []
#     padded_retr_attention_mask = []

#     for item in batch:
#         num_retr = len(item['retr_input_ids_list'])
#         pad_len = max_retr - num_retr
        
#         # Assuming the first retrieved doc's shape is representative for padding
#         pad_ids = torch.zeros_like(item['retr_input_ids_list'][0])
#         pad_mask = torch.zeros_like(item['retr_attention_mask_list'][0])
        
#         padded_ids = item['retr_input_ids_list'] + [pad_ids] * pad_len
#         padded_mask = item['retr_attention_mask_list'] + [pad_mask] * pad_len
        
#         padded_retr_input_ids.append(torch.stack(padded_ids))
#         padded_retr_attention_mask.append(torch.stack(padded_mask))

#     return {
#         'query_input_ids': query_input_ids,
#         'query_attention_mask': query_attention_mask,
#         'retr_input_ids_list': [torch.stack(tensors) for tensors in zip(*padded_retr_input_ids)],
#         'retr_attention_mask_list': [torch.stack(tensors) for tensors in zip(*padded_retr_attention_mask)],
#         'labels': labels
#     }


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    new_tokens = ["<r>", "<c>"]
    tokenizer.add_tokens(new_tokens)

    try:
        with open("data/dataset_dict.json", "r") as f:
            train_dataset_dict = json.load(f)
    except FileNotFoundError:
        print("Error: data/dataset_dict.json not found. Please ensure the path is correct.")
        exit()

    train_dataset = BartDataset(train_dataset_dict, tokenizer, max_length=256)
    
    config = BartConfig.from_pretrained("facebook/bart-base")
    model = CustomBart(config)
    model.resize_token_embeddings(len(tokenizer))
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=8, 
        shuffle=True, 
        # collate_fn=collate_fn
    )

    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    num_epochs = 1
    total_steps = len(train_dataloader) * num_epochs

    progress_bar = tqdm(range(total_steps), desc="Training", leave=True)
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            query_input_ids = batch['query_input_ids'].to(device)
            query_attention_mask = batch['query_attention_mask'].to(device)
            retr_input_ids_list = [ids.to(device) for ids in batch['retr_input_ids_list']]
            retr_attention_mask_list = [mask.to(device) for mask in batch['retr_attention_mask_list']]
            labels = batch['labels'].to(device)

            outputs = model(
                query_input_ids=query_input_ids,
                query_attention_mask=query_attention_mask,
                retr_input_ids_list=retr_input_ids_list,
                retr_attention_mask_list=retr_attention_mask_list,
                labels=labels,
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            progress_bar.update(1)

    progress_bar.close()
    model.save_pretrained('./models/bart_finetuned')