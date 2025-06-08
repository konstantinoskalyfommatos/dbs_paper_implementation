from torch import nn
from transformers import AutoTokenizer, DPRQuestionEncoder
from qdrant_client import QdrantClient

def retrieve_most_similar_docs(
	query: str,
	client: QdrantClient,
	collection_name: str,
	tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base"),
	encoder: nn.Module = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base"),
	limit=5
):
	encoder.to("cuda")

	tokenized_query = tokenizer(
		query,
		add_special_tokens=True,
		return_tensors='pt',
	)
	
	input_ids = tokenized_query['input_ids'].to('cuda')

	embedding_query = encoder(input_ids).pooler_output.squeeze().detach().cpu()

	hits = client.query_points(
		collection_name=collection_name,
		query=embedding_query.tolist(),
		limit=limit,
	).points

	return hits


if __name__ == "__main__":
	client = QdrantClient(
		host="qdrant", 
		port=6333,
	)
	
	COLLECTION_NAME = "e2e_documents_pretrained_dpr"
	query = '<R>name<R>The Hollow Bell Café<R><C><R>familyFriendly<R>no<R><C>'
	# query = 'The Hollow Bell Café'
	# query="cheap, family friendly restaurant"
	hits = retrieve_most_similar_docs(
		query=query,
		client=client,
		collection_name=COLLECTION_NAME,
	)
	print("DONE")
