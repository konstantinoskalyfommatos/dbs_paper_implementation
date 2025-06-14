from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import pandas as pd
from transformers import AutoTokenizer
from torch import nn
from tqdm import tqdm
import torch
import json
from transformers import DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer, DPRContextEncoder


def encode_dataset(
        _df: pd.DataFrame,
        text_col_name: str,
        tokenizer: AutoTokenizer,
        context_encoder: nn.Module,
) -> pd.DataFrame:
    df = _df.copy(deep=True)

    # Get the maximum sentence length to save time during inference
    max_sentence_len = 0
    for sentence in df[text_col_name]:
        input_ids = tokenizer.encode(sentence, add_special_tokens=True)
        max_sentence_len = max(max_sentence_len, len(input_ids))

    context_encoder.to("cuda")
    context_encoder.eval()
    df["embedding"] = None

    # NOTE: https://github.com/huggingface/setfit/issues/287
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating vectors"):
        with torch.no_grad():
            # Tokenize and encode the sentence
            encoded = tokenizer(
                row[text_col_name],
                add_special_tokens=True,
                max_length=max_sentence_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                return_token_type_ids="token_type_ids" in tokenizer.model_input_names,
            )

            input_ids = encoded['input_ids'].to('cuda')

            embedding = context_encoder(input_ids).pooler_output.squeeze().detach().cpu()
            df.at[idx, "embedding"] = embedding
    return df


def main():
    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("models/dpr_finetuned_question_tokenizer")
    context_encoder = DPRContextEncoder.from_pretrained("models/dpr_finetuned_ctx_encoder")

    with open('data/dataset_dict_test.json', 'r') as json_file:
        dataset_dict = json.load(json_file)
    df = pd.DataFrame(dataset_dict.values())
    df.drop(columns=['negative', 'hard_negative', 'ground_truth_retrieved', 'truncated_serialized_query',
                     'serialized_query', 'truncated_serialized_query_csv', 'serialized_query_csv', 'table_str'], inplace=True)
    df = encode_dataset(
        _df=df,
        text_col_name="positive",
        tokenizer=tokenizer,
        context_encoder=context_encoder,
    )

    client = QdrantClient(
        host="localhost",
        port=6333,
    )

    COLLECTION_NAME = "e2e_documents_finetuned_dpr"

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Upserting vectors"):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=idx,
                    vector=row["embedding"].tolist(),
                    payload={
                        key: value for key, value in row.to_dict().items()
                        if key != "embedding"
                    }
                )
            ]
        )

    print("Finished")


if __name__ == "__main__":
    main()
