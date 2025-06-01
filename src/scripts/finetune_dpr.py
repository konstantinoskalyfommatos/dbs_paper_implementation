from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from torch import nn
import random

class DPRDataset(Dataset):
    def __init__(self, data, question_tokenizer, context_tokenizer, max_length=256):
        self.data = data
        self.q_tokenizer = question_tokenizer
        self.ctx_tokenizer = context_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        serialized_query = self._serialize_query(item["table_str"])
        
        positive = item["positive"]
        negs = item.get("negative", [])
        hard_negs = item.get("hard_negative", [])

        negative = negs[0] if negs else None
        hard_negative = hard_negs[0] if hard_negs else None

        query_enc = self.q_tokenizer(serialized_query, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        pos_enc = self.ctx_tokenizer(positive, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        
        if negative:
            neg_enc = self.ctx_tokenizer(negative, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        else:
            neg_enc = None

        if hard_negative:
            hard_neg_enc = self.ctx_tokenizer(hard_negative, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        else:
            hard_neg_enc = None

        return {
            "query_input_ids": query_enc["input_ids"].squeeze(),
            "query_attention_mask": query_enc["attention_mask"].squeeze(),
            "pos_input_ids": pos_enc["input_ids"].squeeze(),
            "pos_attention_mask": pos_enc["attention_mask"].squeeze(),
            "neg_input_ids": neg_enc["input_ids"].squeeze() if neg_enc else None,
            "neg_attention_mask": neg_enc["attention_mask"].squeeze() if neg_enc else None,
            "hard_neg_input_ids": hard_neg_enc["input_ids"].squeeze() if hard_neg_enc else None,
            "hard_neg_attention_mask": hard_neg_enc["attention_mask"].squeeze() if hard_neg_enc else None,
        }
    
    @staticmethod
    def _serialize_query(table: str) -> dict[str, str]:
        """Returns the table as a dictionary.
        
        Example: 'name[The Hollow Bell Café], eatType[restaurant], priceRange[more than £30], familyFriendly[no]'

        # NOTE: 3.1.2 in paper.
        """
        results = []
        table = table.strip()
        
        # Split by comma to get individual attribute-value pairs
        pairs = table.split(', ')
        
        dict_table = {}
        for pair in pairs:
            key = pair.split("[")[0].strip()
            value = pair.split("[")[1].replace("]", "").strip()
            dict_table[key] = value

        dict_table[random.choice([key for key in dict_table.keys() if key != 'name'])] = ''
        del dict_table[random.choice([key for key in dict_table.keys() if key != 'name'])]

        for key, value in dict_table.items():
            results.append(f"<R>{key}<R>{value}<R>")

        return '<C>'.join(results)


class DPRTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        q_inputs = {"input_ids": inputs["query_input_ids"], "attention_mask": inputs["query_attention_mask"]}
        p_inputs = {"input_ids": inputs["pos_input_ids"], "attention_mask": inputs["pos_attention_mask"]}

        q_embeds = model.question_encoder(**q_inputs).pooler_output
        p_embeds = model.ctx_encoder(**p_inputs).pooler_output

        # Compute similarities (batch dot products)
        scores = torch.matmul(q_embeds, p_embeds.T)  # shape: (B, B)
        labels = torch.arange(len(scores)).to(scores.device)
        loss = F.cross_entropy(scores, labels)
        
        return (loss, scores) if return_outputs else loss


question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")


class DPRDualEncoder(nn.Module):
    def __init__(self, question_encoder, ctx_encoder):
        super().__init__()
        self.question_encoder = question_encoder
        self.ctx_encoder = ctx_encoder

    def forward(self):
        # Dummy forward (not used by Trainer)
        return None


training_args = TrainingArguments(
    output_dir="./dpr_finetuned",
    per_device_train_batch_size=512,
    learning_rate=1e-5,
    num_train_epochs=2,
    logging_dir='./logs',
    save_strategy="epoch"
)

trainer = DPRTrainer(
    model=DPRDualEncoder(question_encoder, ctx_encoder),
    args=training_args,
    train_dataset=your_dataset,
    tokenizer=q_tokenizer,  # optional
)
trainer.train()


# query_embedding = question_encoder(**q_tokenizer("your query", return_tensors="pt")).pooler_output
# ctx_embedding = ctx_encoder(**ctx_tokenizer("some passage", return_tensors="pt")).pooler_output
