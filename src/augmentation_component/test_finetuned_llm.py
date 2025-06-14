from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def format_inference_prompt(query_table, documents, tokenizer):
    system_prompt = (
        "You are an expert in text to table. The user will provide a query table and a list of retrieved documents that may or may not be related to the query. The query may be missing values and/or attributes. Your task is to:\n"
        "- Decide which documents can be used based on the query.\n"
        "- Fill the missing values *always* based on the documents.\n"
        "- Add extra attributes and their corresponding values, if needed, *always* based on the documents.\n"
        "- The available attributes are: name, priceRange, eatType, familyFriendly, near, customer rating, food, area"
    )

    chat_template = """<|im_start|>system
{}
<|im_end|>
<|im_start|>user
Query: {}

Documents: {}
<|im_end|>
<|im_start|>assistant
"""

    documents_text = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(documents)])
    prompt = chat_template.format(system_prompt, query_table, documents_text)

    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    return inputs

# Path to your merged model (should contain config.json, model weights, tokenizer files, etc.)
finetuned_lora_model_dir_16_bit = "/home/giorgos/Documents/MSc/databases/Project/dbs_proj_retrieval_component/models/finetuned_lora_model_16_bit"

# Initialize the vLLM model
llm = LLM(model=finetuned_lora_model_dir_16_bit, dtype="float16")  # You can also try "auto"

# Sampling parameters (adjust as needed)
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,  # adjust based on expected output length
)


# Function to generate response
def generate_response(query_table, documents, tokenizer: AutoTokenizer):
    prompt = format_inference_prompt(query_table, documents, tokenizer)
    input_text = tokenizer.decode(prompt['input_ids'][0], skip_special_tokens=True)

    outputs = llm.generate([input_text], sampling_params)
    return outputs[0].outputs[0].text.strip()


# Example usage
query_table = "name|priceRange|familyFriendly|\nThe Hollow Bell Café|below 20|"
documents = [
    "If you are looking for an inexpensive, family friendly restaurant, The Hollow Bell Café is the place to go.",
    # "The The Hollow Bell Café is not children friendly cost more than £30.",
    "A family friendly restaurant, The Hollow Bell Café, is not expensive.",
    "The The Hollow Bell Café is an adult only cheat restaurant.",
    "If you are looking for an inexpensive, family friendly restaurant, The Hollow Bell Café is the place to go."
]
tokenizer = AutoTokenizer.from_pretrained(finetuned_lora_model_dir_16_bit)
response = generate_response(query_table, documents, tokenizer)
print("Model output:\n", response)
