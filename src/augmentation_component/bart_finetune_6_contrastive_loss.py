from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
import json
from tqdm import tqdm


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right for teacher forcing.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("pad_token_id has to be defined.")
    # replace possible -100 values in labels by pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class BartDataset(Dataset):
    def __init__(self, data: dict, tokenizer, max_query_length=128, max_doc_length=256, k=5):
        self.data_dict = data
        self.data_keys = sorted(data.keys(), key=int)
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.k = k

    def __len__(self):
        return len(self.data_keys)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item_key = self.data_keys[idx]
        item = self.data_dict[item_key]

        retrieved_documents = item.get("ground_truth_retrieved", []) + [item.get("positive", "")]

        if len(retrieved_documents) > self.k:
            retrieved_documents = retrieved_documents[:self.k]
        else:
            retrieved_documents += [""] * (self.k - len(retrieved_documents))

        query_enc = self.tokenizer(
            item["truncated_serialized_query"],
            truncation=True,
            padding="max_length",
            max_length=self.max_query_length,
            return_tensors="pt"
        )

        retr_encs = self.tokenizer(
            retrieved_documents,
            truncation=True,
            padding="max_length",
            max_length=self.max_doc_length,
            return_tensors="pt"
        )

        labels = self.tokenizer(
            item["table_str"],
            truncation=True,
            padding="max_length",
            max_length=self.max_query_length,
            return_tensors="pt"
        )
        # Set padding tokens to -100 so they are ignored in the loss function
        labels["input_ids"][labels["input_ids"] == self.tokenizer.pad_token_id] = -100

        return {
            "query_input_ids": query_enc["input_ids"].squeeze(0),
            "query_attention_mask": query_enc["attention_mask"].squeeze(0),
            "retr_input_ids": retr_encs["input_ids"],
            "retr_attention_mask": retr_encs["attention_mask"],
            "labels": labels["input_ids"].squeeze(0)
        }


# --- FIX 1: Inherit from BartForConditionalGeneration ---
# This class provides the necessary `.generate()` method and architecture.
class BartWithKnnInterpolation(BartForConditionalGeneration):
    def __init__(self, config, contrastive_loss_ratio=0.1, temperature=0.1):
        # --- FIX 2: Call the parent __init__ ---
        # This properly initializes the underlying BART model (self.model) and the LM head (self.lm_head).
        super().__init__(config)
        # Define only the additional layers needed for your custom logic.
        self.proj_q = nn.Linear(config.d_model, config.d_model)
        self.proj_d = nn.Linear(config.d_model, config.d_model)
        self.contrastive_loss_ratio = contrastive_loss_ratio
        self.temperature = temperature


    # --- FIX 3: Restructure the `forward` pass ---
    # The `forward` method is updated to handle two cases:
    # 1. A full pass (encoding + decoding) for training/evaluation.
    # 2. A decoder-only pass for the iterative steps within `.generate()`.
    def forward(
            self,
            query_input_ids=None,
            query_attention_mask=None,
            retr_input_ids=None,
            retr_attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            attention_mask=None,  # This is used for the cross-attention in the decoder
            labels=None,
            use_cache=None,
            return_dict=None,
            **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        if encoder_outputs is None:
            # This block executes during training or the first step of generation.
            # It runs the custom logic to create the interpolated encoder representation.
            # 1. ENCODE QUERY
            query_encoder_outputs = self.model.encoder(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
                return_dict=True
            )
            T_e = query_encoder_outputs.last_hidden_state[:, 0, :]

            # 2. ENCODE DOCUMENTS
            batch_size, k, doc_len = retr_input_ids.shape
            hidden_size = T_e.shape[-1]

            docs_input_ids_flat = retr_input_ids.view(-1, doc_len)
            docs_attention_mask_flat = retr_attention_mask.view(-1, doc_len)

            doc_encoder_outputs = self.model.encoder(
                input_ids=docs_input_ids_flat,
                attention_mask=docs_attention_mask_flat,
                return_dict=True
            )
            d_e_flat = doc_encoder_outputs.last_hidden_state[:, 0, :]
            d_e = d_e_flat.view(batch_size, k, hidden_size)

            # 3. LINEAR PROJECTIONS
            T_p = self.proj_q(T_e)
            d_p = self.proj_d(d_e)


            # 4. COMPUTE INTERPOLATION WEIGHTS
            T_p_unsqueezed = T_p.unsqueeze(1)
            query_doc_similarity = (T_p_unsqueezed * d_p).sum(dim=-1)
            doc_doc_similarity_matrix = torch.matmul(d_p, d_p.transpose(1, 2))
            doc_doc_similarity_sum = doc_doc_similarity_matrix.sum(dim=-1) - torch.diagonal(doc_doc_similarity_matrix,
                                                                                            dim1=-2, dim2=-1)
            total_scores = query_doc_similarity + doc_doc_similarity_sum
            w = F.softmax(total_scores, dim=-1)

            # 5. WEIGHTED AVERAGE AND CONCATENATION
            w_unsqueezed = w.unsqueeze(-1)
            weighted_avg_docs = (w_unsqueezed * d_e).sum(dim=1)
            interpolated_hidden_states = torch.stack([T_e, weighted_avg_docs], dim=1)

            # This is the attention mask for the decoder's cross-attention.
            # It has a length of 2, corresponding to [query_embedding, doc_embedding].
            attention_mask = torch.ones(interpolated_hidden_states.shape[:2], device=self.device)

            encoder_outputs = BaseModelOutput(
                last_hidden_state=interpolated_hidden_states,
                hidden_states=None,
                attentions=None,
            )

        # For training, create `decoder_input_ids` from `labels` if not provided
        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # --- DECODER ---
        # This part is standard and is now compatible with the outputs from the custom encoder logic.
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            # Combine query and document projections for contrastive loss calculation
            # The shape will be (batch_size, k + 1, hidden_size)
            all_projections = torch.cat([T_p.unsqueeze(1), d_p], dim=1)

            # Normalize the projections
            all_projections = F.normalize(all_projections, p=2, dim=2)

            # Calculate cosine similarity
            cos_sim = torch.matmul(all_projections, all_projections.transpose(1, 2)) / self.temperature

            # Create masks to identify positive and negative pairs
            # Positive pairs are (query, doc_i) for all i
            # and (doc_i, query) for all i
            mask = torch.zeros_like(cos_sim)
            mask[:, 0, 1:] = 1
            mask[:, 1:, 0] = 1

            # Use log_softmax for numerical stability
            log_prob = F.log_softmax(cos_sim, dim=2)

            # Calculate the mean of the log probabilities of the positive pairs
            # Negative log-likelihood is used, so we take the negative of the sum
            contrastive_loss = - (log_prob * mask).sum(dim=2).sum(dim=1) / mask.sum(dim=2).sum(dim=1)
            contrastive_loss = contrastive_loss.mean()

            loss_fct = nn.CrossEntropyLoss()
            seq2seq_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

            loss = seq2seq_loss + self.contrastive_loss_ratio * contrastive_loss

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        )


def main():
    # 1. INITIALIZE TOKENIZER AND MODEL
    model_name = 'facebook/bart-large'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartWithKnnInterpolation.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    dataset_filename = "data/dataset_dict.json"
    try:
        with open(dataset_filename, "r") as f:
            train_dataset_dict = json.load(f)
    except FileNotFoundError:
        print(f"Error: {dataset_filename} not found.")
        exit()

    K_DOCS = 4
    dataset = BartDataset(
        train_dataset_dict,
        tokenizer,
        max_query_length=128,
        max_doc_length=256,
        k=5  # Set your desired number of documents
    )

    # 3. CREATE DATALOADER
    dataloader = DataLoader(dataset, batch_size=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # 4. TRAINING LOOP
    print("\n--- Training Loop Example ---")
    num_of_epochs = 2
    model.train()
    for epoch in range(num_of_epochs):
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)*num_of_epochs):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            print(f"Computed training loss: {loss.item():.4f}")
            loss.backward()
            optimizer.step()
            if idx % 100 == 0 and idx > 0:
                model.save_pretrained('./models/bart_finetuned')

    del model
    model = BartWithKnnInterpolation.from_pretrained('./models/bart_finetuned')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 5. INFERENCE/GENERATION
    print("\n--- Inference Example ---")
    model.eval()
    with torch.no_grad():
        inference_batch = next(iter(dataloader))
        inference_batch = {k: v.to(device) for k, v in inference_batch.items()}

        # --- FIX 4: The `.generate()` call now works correctly ---
        # The arguments are passed to our custom `forward` method, which creates
        # the interpolated encoder state before generation begins.
        generated_ids = model.generate(
            query_input_ids=inference_batch["query_input_ids"],
            query_attention_mask=inference_batch["query_attention_mask"],
            retr_input_ids=inference_batch["retr_input_ids"],
            retr_attention_mask=inference_batch["retr_attention_mask"],
            num_beams=4,
            max_length=50,  # Reduced max_length for faster generation in this example
            early_stopping=True
        )

        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        original_queries = tokenizer.batch_decode(inference_batch["query_input_ids"], skip_special_tokens=True)

        # Decode labels, making sure to replace -100 with the pad token ID before decoding
        labels_for_decoding = inference_batch["labels"].clone()
        labels_for_decoding[labels_for_decoding == -100] = tokenizer.pad_token_id
        original_labels = tokenizer.batch_decode(labels_for_decoding, skip_special_tokens=True)

        for i in range(len(generated_texts)):
            print(f"\nQuery {i + 1}: '{original_queries[i]}'")
            print(f"Expected Output: '{original_labels[i]}'")
            print(f"Generated Output: '{generated_texts[i]}'")
            print("-" * 20)


if __name__ == '__main__':
    main()
