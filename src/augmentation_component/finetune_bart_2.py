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
    def __init__(self, data: dict, tokenizer, max_query_length=128, max_doc_length=256, k=5):
        self.data_dict = data
        self.data_keys = sorted(data.keys(), key=int)
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.k = k # Number of documents to use

    def __len__(self):
        return len(self.data_keys)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item_key = self.data_keys[idx]
        item = self.data_dict[item_key]
        
        # Ensure there's always at least one positive example
        retrieved_documents = item.get("ground_truth_retrieved", []) + [item.get("positive", "")]
        
        # --- FIX: Pad or truncate the list of documents to a fixed size k ---
        if len(retrieved_documents) > self.k:
            retrieved_documents = retrieved_documents[:self.k]
        else:
            # Pad with empty strings if there are fewer than k documents
            retrieved_documents += [""] * (self.k - len(retrieved_documents))

        query_enc = self.tokenizer(
            item["truncated_serialized_query"],
            truncation=True,
            padding="max_length",
            max_length=self.max_query_length,
            return_tensors="pt"
        )

        # Tokenize all k documents
        retr_encs = self.tokenizer(
            retrieved_documents,
            truncation=True,
            padding="max_length",
            max_length=self.max_doc_length,
            return_tensors="pt"
        )

        labels = self.tokenizer(
            item["table_str"], # Use table_str as the ground truth for generation
            truncation=True,
            padding="max_length",
            max_length=self.max_query_length, # Labels are often similar in length to the query
            return_tensors="pt"
        )

        return {
            "query_input_ids": query_enc["input_ids"].squeeze(0),
            "query_attention_mask": query_enc["attention_mask"].squeeze(0),
            # --- FIX: Return stacked tensors instead of a list ---
            "retr_input_ids": retr_encs["input_ids"],
            "retr_attention_mask": retr_encs["attention_mask"],
            "labels": labels["input_ids"].squeeze(0)
        }

class CustomBart(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.proj_query = nn.Linear(config.d_model, config.d_model // 2)
        self.proj_retr = nn.Linear(config.d_model, config.d_model // 2)

    def _get_combined_encoder_outputs(
        self,
        query_input_ids: torch.LongTensor,
        query_attention_mask: torch.Tensor,
        retr_input_ids: torch.LongTensor,
        retr_attention_mask: torch.Tensor,
        return_dict: Optional[bool] = None
    ):
        batch_size, k, doc_seq_len = retr_input_ids.shape
        query_seq_len = query_input_ids.shape[1]

        # 1. Encode query
        query_outputs = self.model.encoder(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            return_dict=return_dict,
        )
        query_encoder_hidden_states = query_outputs[0] # (batch, query_seq_len, d_model)

        # 2. Encode documents
        # Reshape for batch encoding: (batch * k, doc_seq_len)
        retr_input_ids_flat = retr_input_ids.view(batch_size * k, doc_seq_len)
        retr_attention_mask_flat = retr_attention_mask.view(batch_size * k, doc_seq_len)

        retr_outputs_flat = self.model.encoder(
            input_ids=retr_input_ids_flat,
            attention_mask=retr_attention_mask_flat,
            return_dict=return_dict,
        )[0] # (batch * k, doc_seq_len, d_model)

        # Reshape back: (batch, k, doc_seq_len, d_model)
        d_enc = retr_outputs_flat.view(batch_size, k, doc_seq_len, self.config.d_model)

        # 3. Project encoder outputs
        # For simplicity, we take the mean over the sequence length dimension to get a single vector per query/document.
        # This is a common practice and simplifies the dot products significantly.
        T_hat = self.proj_query(query_encoder_hidden_states.mean(dim=1)) # (batch, d_model/2)
        d_hat = self.proj_retr(d_enc.mean(dim=2)) # (batch, k, d_model/2)

        # 4. Calculate interpolation weights (Vectorized)
        # T_hat · d_hat_i
        query_doc_similarity = torch.einsum('bd,bkd->bk', T_hat, d_hat) # (batch, k)

        # d_hat_i · d_hat_j
        doc_doc_products = torch.einsum('bid,bjd->bij', d_hat, d_hat) # (batch, k, k)
        
        # Sum over j and subtract the diagonal (i=j case)
        doc_doc_similarity = doc_doc_products.sum(dim=2) - torch.diagonal(doc_doc_products, dim1=-2, dim2=-1)
        
        interpolation_scores = query_doc_similarity + doc_doc_similarity
        softmax_weights = F.softmax(interpolation_scores, dim=1) # (batch, k)

        # 5. Compute weighted average of document encoder outputs
        # Use unsqueeze to make weights broadcast-able: (batch, k, 1, 1)
        weights_expanded = softmax_weights.unsqueeze(-1).unsqueeze(-1)
        weighted_doc_outputs = (d_enc * weights_expanded).sum(dim=1) # (batch, doc_seq_len, d_model)

        # 6. Concatenate query and weighted documents for the decoder
        # Note: According to the paper, the final length is max_query + max_doc
        combined_encoder_outputs = torch.cat((query_encoder_hidden_states, weighted_doc_outputs), dim=1)
        
        # Create the corresponding attention mask
        # The weighted doc output has the structure of a single document
        weighted_doc_attention_mask = torch.ones(
            weighted_doc_outputs.shape[:2],
            dtype=torch.long,
            device=weighted_doc_outputs.device
        )
        combined_attention_mask = torch.cat((query_attention_mask, weighted_doc_attention_mask), dim=1)
        
        # Wrap in a BaseModelOutput for compatibility with the decoder
        return_val = self.model.encoder.get_output_return_dict(
            last_hidden_state=combined_encoder_outputs,
            outputs=query_outputs # Base it on query outputs
        )
        return return_val, combined_attention_mask

    def forward(
        self,
        query_input_ids: torch.LongTensor,
        query_attention_mask: torch.Tensor,
        retr_input_ids: torch.LongTensor,
        retr_attention_mask: torch.Tensor,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:

        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # --- FIX for Inference: Use pre-computed encoder_outputs if available ---
        if encoder_outputs is None:
            encoder_outputs, encoder_attention_mask = self._get_combined_encoder_outputs(
                query_input_ids=query_input_ids,
                query_attention_mask=query_attention_mask,
                retr_input_ids=retr_input_ids,
                retr_attention_mask=retr_attention_mask,
                return_dict=True
            )
        else:
            # During generate(), the attention mask is passed separately
            encoder_attention_mask = kwargs.get("attention_mask")


        # The rest of the forward pass is standard BartForConditionalGeneration
        return super().forward(
            input_ids=None, # input_ids is not used when encoder_outputs is provided
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            labels=labels,
            **kwargs,
        )

    # --- FIX for Inference: Custom generate method ---
    @torch.no_grad()
    def generate(
        self,
        query_input_ids: torch.LongTensor,
        query_attention_mask: torch.Tensor,
        retr_input_ids: torch.LongTensor,
        retr_attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.LongTensor:
        self.eval() # Ensure model is in eval mode

        encoder_outputs, encoder_attention_mask = self._get_combined_encoder_outputs(
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            retr_input_ids=retr_input_ids,
            retr_attention_mask=retr_attention_mask,
            return_dict=True
        )

        # Call the original generate method with our custom encoder state
        return super().generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            **kwargs
        )


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    # Using the special tokens from your dataset context
    new_tokens = ["<r>", "<c>"]
    tokenizer.add_tokens(new_tokens)

    try:
        with open("data/dataset_dict.json", "r") as f:
            train_dataset_dict = json.load(f)
    except FileNotFoundError:
        print("Error: data/dataset_dict.json not found.")
        exit()

    # --- Use the updated Dataset with separate max lengths and k ---
    # You should analyze your dataset to find optimal values for these
    train_dataset = BartDataset(
        train_dataset_dict, 
        tokenizer, 
        max_query_length=128, 
        max_doc_length=256, 
        k=5 # Set your desired number of documents
    )
    
    config = BartConfig.from_pretrained("facebook/bart-base")
    model = CustomBart(config)
    model.resize_token_embeddings(len(tokenizer))
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=4, # Lower batch size might be needed due to larger memory footprint
        shuffle=True, 
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
            
            # --- FIX: Update keys to match new dataset output ---
            outputs = model(
                query_input_ids=batch['query_input_ids'].to(device),
                query_attention_mask=batch['query_attention_mask'].to(device),
                retr_input_ids=batch['retr_input_ids'].to(device),
                retr_attention_mask=batch['retr_attention_mask'].to(device),
                labels=batch['labels'].to(device),
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            progress_bar.update(1)

    progress_bar.close()
    model.save_pretrained('./models/bart_finetuned_knn')
    tokenizer.save_pretrained('./models/bart_finetuned_knn')

    # --- Example of how to use the custom generate method ---
    print("\n--- Running Inference Example ---")
    model.eval()
    sample_batch = next(iter(train_dataloader))
    
    generated_ids = model.generate(
        query_input_ids=sample_batch['query_input_ids'].to(device),
        query_attention_mask=sample_batch['query_attention_mask'].to(device),
        retr_input_ids=sample_batch['retr_input_ids'].to(device),
        retr_attention_mask=sample_batch['retr_attention_mask'].to(device),
        num_beams=4,
        max_length=150,
        early_stopping=True
    )
    
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    for i, text in enumerate(generated_text):
        print(f"Sample {i+1} Input Query: {tokenizer.decode(sample_batch['query_input_ids'][i], skip_special_tokens=True)}")
        print(f"Sample {i+1} Generated Table: {text}")
        print("-" * 20)