from vllm import LLM, SamplingParams
from dotenv import load_dotenv
import os
import ast
import re

# load_dotenv()
# token = os.getenv("HUGGINGFACE_HUB_TOKEN")
# if token:
#     login(token)

llm = LLM("Qwen/Qwen3-1.7B")

sampling_params = SamplingParams(
    temperature=0.7,
    top_k=20,
    top_p=0.8,
    max_tokens=8192
)

GENERATE_RESTAURANT_NAMES_PROMPT = """/no_think Your task is to generate additional, unique shop names.

Do not number them. Do not include anything else.

Just output a Python list of strings, wrapped between <list> and </list>.

Example:
<list>["The Rustic Spoon", "Midnight Roastery", "Fog & Bean", ...]</list>

A shop can be a restaurant, coffee shop, or pub.

Existing names: {}
"""

with open("data/unique_names.txt", "r") as f:
    names = [line.strip() for line in f if line.strip()]

existing_names = set(names)
errors = 0
patience_threshold = 20

while len(names) < 100 and errors < patience_threshold:
    prompt = GENERATE_RESTAURANT_NAMES_PROMPT.format(names)

    outputs = llm.generate([prompt], sampling_params=sampling_params)

    for output in outputs:
        text = output.outputs[0].text.strip()
        match = re.search(r"<list>(.*?)</list>", text, re.DOTALL)

        if not match:
            print("No <list> tag found. Output was:")
            print(text)
            errors += 1
            continue

        list_str = match.group(1).strip()

        try:
            parsed_names = ast.literal_eval(f"[{list_str}]" if not list_str.startswith("[") else list_str)
            new_names = [name for name in parsed_names if isinstance(name, str) and name not in existing_names]

            if not new_names:
                errors += 1
                print("No new valid names found.")
                continue

            names.extend(new_names)
            existing_names.update(new_names)
            print(f"Added {len(new_names)} new names. Total: {len(names)}")

        except Exception as e:
            print(f"Parsing failed: {e}")
            errors += 1

with open("data/synthetic_names.txt", "w") as f:
    for idx, name in enumerate(names, start=1):
        f.write("\n".join(names))

print("Done. Names written to data/synthetic_names.txt")
