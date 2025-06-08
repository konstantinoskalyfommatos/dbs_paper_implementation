from vllm import LLM, SamplingParams
from dotenv import load_dotenv
from transformers import AutoTokenizer
import os
import ast
import re
import numpy as np
from huggingface_hub import login

load_dotenv()
token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if token:
    login(token)

def get_formatted_prompt(tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"), prompt_factor: int = 20):
	SHOP_NAMES = [
		# Coffee Shops (50)
		"Bean & Gone", "Brewed Awakening", "The Daily Grind", "Perk Up", "Kaffeine Fix",
		"Steamy Beans", "Espresso Lane", "Mocha Muse", "Latte Love", "The Roasted Bean",
		"Java Junction", "Caffeine Alley", "The Grind House", "Cup o’ Joy", "Drip Theory",
		"Flat White & Co", "Sip Society", "The Cozy Mug", "Urban Brew", "Café Velvet",
		"Morning Ritual", "Brewhouse Café", "Cuppa Cloud", "Artisan Bean", "The Pour Over",
		"Steam Theory", "The Java Nook", "Coffee Republic", "Latte Lounge", "The Drip Stop",
		"Midnight Mocha", "Wake Up Café", "Second Cup", "Mug Life", "The Roastery",
		"Daily Drip", "Coffee Bloom", "Crimson Cup", "Bitter & Sweet", "Fuel & Filter",
		"The Buzzing Bean", "Luna Latte", "Daily Dose", "Ground Control", "The Roasted Leaf",
		"Whisk & Bean", "Black Gold", "Cozy Cup", "The Bean Scene", "Brew Social", "Cloud Nine Café",

		# Restaurants (50)
		"The Rice Boat", "Spice Symphony", "Golden Fork", "Olive & Thyme", "The Hungry Fox",
		"Fork & Fable", "Little Lemon", "Saffron Table", "Salt & Fire", "The Green Olive",
		"Crimson Curry", "Rustic Spoon", "Harvest Moon", "Blue Basil", "Ginger Flame",
		"Luna Kitchen", "The Silver Chopstick", "Midnight Diner", "The Wandering Chef", "The Charred Oak",
		"Urban Palate", "Tandoori Tales", "Maison Verde", "Fig & Fennel", "Bamboo Bowl",
		"Cedar & Sage", "The Broken Plate", "Fried & True", "Nomad’s Nosh", "The Food Foundry",
		"Peach & Pepper", "Hidden Fork", "The Gilded Wok", "The Wholesome Fork", "Plate & Pour",
		"Wild Thyme", "Juniper Table", "The Brass Onion", "Coastal Cravings", "The Silk Table",
		"Crave Street", "Spork & Spoon", "The Marble Kitchen", "Fire & Ice", "The Curry Leaf",
		"Amalfi Kitchen", "Noodle & Bone", "The Spicy Mango", "Mezza Luna", "The Tasty Turnip", "Crust & Crumble",

		# Pubs (50)
		"The Drunken Duck", "The Tipsy Crow", "The Laughing Pint", "The Thirsty Raven", "The Blind Donkey",
		"The Crooked Antler", "The Rusty Tap", "The Jolly Badger", "The Howling Wolf", "The Black Boar",
		"The Lazy Otter", "The Old Barrel", "The Drunken Stag", "The Wandering Mule", "The Dapper Fox",
		"The Stout & Crown", "The Pickled Herring", "The Cursed Hound", "The Amber Keg", "The Gilded Goat",
		"The Wicked Ale", "The Salty Dog", "The Dancing Tankard", "The Broken Stein", "The Brass Tap",
		"The Grinning Cat", "The Bitter End", "The Hidden Flask", "The Clover & Oak", "The Cozy Tankard",
		"The Muddy Boot", "The Sleepy Bear", "The Wandering Barrel", "The Noble Fir", "The Twisted Hop",
		"The Thirsty Troll", "The Tipsy Wren", "The Grumpy Monk", "The Blue Ox", "The Wandering Elk",
		"The Rusted Crown", "The Lucky Duck", "The Hoppy Fox", "The Kraken’s Cup", "The Red Lantern",
		"The Midnight Mug", "The Grog & Grub", "The Rowdy Raven", "The Broken Horn", "The Lost Lantern", "The Sly Toad"
	]

	GENERATE_RESTAURANT_NAMES_PROMPT_TEMPLATE = """You are a creative shop name suggester. Your task is to write a deduplicated python list of fifty (50) unique shop names.

	Do not provide any additional information; just output a Python list of names.

	A shop can be a restaurant, coffee shop, or pub.

	Example names: {}, {}, {}.
	"""

	prompts = []
	for _ in range(prompt_factor):
		chosen_names = np.random.choice(SHOP_NAMES, replace=False, size=3)
		
		messages = [
			{
				"role": "user", 
				"content": GENERATE_RESTAURANT_NAMES_PROMPT_TEMPLATE.format(
					chosen_names[0], 
					chosen_names[1], 
					chosen_names[2]
				)
			}
		]
		prompt = tokenizer.apply_chat_template(
			messages,
			tokenize=False,
			add_generation_prompt=True,
			enable_thinking=False
		)
		prompts.append(prompt)
	return prompts


def main():
	prompt_factor = 20

	print("Loading model")
	llm = LLM(
		"Qwen/Qwen3-4B",
		max_model_len=2000,
		max_num_seqs=prompt_factor,
		dtype="auto",
		trust_remote_code=True,
		quantization="fp8"
	)
	print("Loaded model")
	tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

	sampling_params = SamplingParams(
		temperature=0.7,
		top_k=20,
		top_p=0.8,
		max_tokens=3000
	)

	with open("data/unique_names.txt", "r") as f:
		original_names = [line.strip() for line in f if line.strip()]

	names = original_names
	TARGET_NAME_COUNT = 10000
	try:
		while len(names) < TARGET_NAME_COUNT + len(original_names):
			print(len(names))
			outputs = llm.generate(
				get_formatted_prompt(
					tokenizer=tokenizer,
					prompt_factor=prompt_factor
				), 
				sampling_params=sampling_params
			)

			for output in outputs:
				text = output.outputs[0].text.strip()

				match = re.search(r"\[(.*?)\]", text, re.DOTALL)

				if not match:
					print("No list found. Output was:")
					print(text)
					continue

				list_str = match.group(1).strip()

				try:
					parsed_names = ast.literal_eval(list_str)
					names.extend(parsed_names)
					names = list(set(names))
				except Exception as e:
					print(f"Parsing failed: {e}")

	except Exception as e:
		print(f"Error: {e}")

	finally:
		names = list(set(names))
		for orig_name in original_names:
			if orig_name in names:
				names.remove(orig_name)
		with open("data/synthetic_names.txt", "w") as f:
			f.write("\n".join(names))

	print("Done. Names written to data/synthetic_names.txt")


if __name__ == "__main__":
	main()