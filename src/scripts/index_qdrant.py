from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import pandas as pd
from transformers import AutoTokenizer, DPRContextEncoder
from torch import nn
from tqdm import tqdm
import torch


def encode_dataset(
	_df: pd.DataFrame, 
	text_col_name: str,
	tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base"),
	encoder: nn.Module = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base"),
) -> pd.DataFrame:
	
	df = _df.copy(deep=True)

	# Get the maximum sentence length to save time during inference
	max_sentence_len = 0
	for sentence in df[text_col_name]:
		input_ids = tokenizer.encode(sentence, add_special_tokens=True)
		max_sentence_len = max(max_sentence_len, len(input_ids))

	encoder.to("cuda")
	encoder.eval()
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

			embedding = encoder(input_ids).pooler_output.squeeze().detach().cpu()
			df.at[idx, "embedding"] = embedding
	return df


def main():
	df = pd.read_csv("/home/retrieval_component/data/new_train.csv")
	df.drop(columns=['orig_mr', 'Original Name', 'ref', 'fixed'], inplace=True)
	df = encode_dataset(
		_df=df,
		text_col_name="synthetic ref"
	)

	client = QdrantClient(
		host="qdrant", 
		port=6333,
	)

	COLLECTION_NAME = "e2e_documents_pretrained_dpr"

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
