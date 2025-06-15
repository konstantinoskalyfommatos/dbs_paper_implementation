import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from argparse import ArgumentParser


SYSTEM_PROMPT = (
    "You are a helpful assistant that only responds with a formatted pipe-separated table wrapped in <table> </table> tags. "
    "The user provides a query table in pipe-separated format and a collection of available documents. "
    "Your task is to fill missing values or/and add additional attributes with their corresponding values based *only* on the available documents. "
    "Ignore unrelated documents. The available attributes are: [name, priceRange, eatType, familyFriendly, near, customer rating, food, area]."
)

USER_PROMPT = """##### Query:
<table>{}</table>

##### Documents:
# {}"""


def format_prompts(
    query_table: str, 
    documents: list[str], 
    tokenizer,
) -> str:
    """Formats the query table and documents into a structured prompt for the model."""
    documents_text = "\n".join([f"{i + 1}) {doc}" for i, doc in enumerate(documents)])

    messages = [
        # NOTE: Gemma3 is not trained on system prompts. it will automatically add it in the user prompt
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(query_table, documents_text)},
    ]

    formatted_conv = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True,
    )
    return formatted_conv


def main():
    """Main function to load data, run inference, and save results."""

    parser = ArgumentParser()

    parser.add_argument(
        "--test_set",
        action="store_true",
        help="When True, predictions will be made for the test set's ground truth documents"
    )
    args = parser.parse_args()

    if args.test_set:
        print("Running predictions for the test set's ground truth documents.")
        DATASET_DICT_PATH = "data/dataset_dict_test.json"
        OUTPUT_JSON_PATH = "data/results_ground_truth.json"
    else:
        # Predictions for ducuments retrieved from DPR index, where s2e documents are used
        print("Running predictions for the DPR index documents.")
        DATASET_DICT_PATH = "data/dataset_dict_test_e2e.json"
        OUTPUT_JSON_PATH = "data/results_e2e.json"
    
    result_dict = {}
    
    with open(DATASET_DICT_PATH, "rb") as f:
        dataset_dict: dict[str, dict] = json.load(f)

    finetuned_lora_model_dir_16_bit = "models/finetuned_gemma-3-1b-it-unsloth-bnb-4bit-lora-16bit"

    llm = LLM(
        model=finetuned_lora_model_dir_16_bit,
        dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(finetuned_lora_model_dir_16_bit)
    
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        max_tokens=150, 
        top_k=64
    )

    data_indices = list(dataset_dict.keys()) 
    batch_prompts = []

    for idx in data_indices:
        data_point = dataset_dict[idx]
        
        result_dict[idx] = {
            "truncated_serialized_query_csv": data_point["truncated_serialized_query_csv"],
            "serialized_query_csv": data_point["serialized_query_csv"],
            "ground_truth_retrieved": data_point["ground_truth_retrieved"],
            "enriched_truncated_serialized_query_csv": ""  # To be filled
        }
        
        formatted_prompt = format_prompts(
            query_table=data_point["truncated_serialized_query_csv"],
            documents=data_point["ground_truth_retrieved"],
            tokenizer=tokenizer,
        )
        batch_prompts.append(formatted_prompt)

    
    outputs = llm.generate(batch_prompts, sampling_params)
    

    for i, output in enumerate(outputs):
        original_idx = data_indices[i]
        generated_text = output.outputs[0].text.split("<table>")[1].split("</table")[0].strip()
        result_dict[original_idx]["enriched_truncated_serialized_query_csv"] = generated_text

    print(f"Saving results to {OUTPUT_JSON_PATH}")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)
        
    print("Processing complete. Results have been saved.")


if __name__ == "__main__":
    main()
