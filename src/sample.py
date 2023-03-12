""" Sample code for running the BigBird for Conditional Generation """

from transformers import PegasusTokenizer, BigBirdPegasusForConditionalGeneration
import torch
"""
tokenizer = AutoTokenizer.from_pretrained(
    "google/bigbird-pegasus-large-bigpatent")

# by default encoder-attention is `block_sparse` with num_random_blocks=3, block_size=64
model = BigBirdPegasusForConditionalGeneration.from_pretrained(
    "google/bigbird-pegasus-large-bigpatent")

# decoder attention type can't be changed & will be "original_full"
# you can change `attention_type` (encoder only) to full attention like this:
model = BigBirdPegasusForConditionalGeneration.from_pretrained(
    "google/bigbird-pegasus-large-bigpatent", attention_type="original_full")

# you can change `block_size` & `num_random_blocks` like this:
model = BigBirdPegasusForConditionalGeneration.from_pretrained(
    "google/bigbird-pegasus-large-bigpatent", block_size=16, num_random_blocks=2)

print("Loading in dataset...")
#multi_lexsum = load_dataset("allenai/multi_lexsum", name="v20220616")
multi_lexsum = load_dataset("big_patent")
"""
"""
tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
# model = BigBirdForPreTraining.from_pretrained("google/bigbird-roberta-base")
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    "google/bigbird-roberta-base", "google/bigbird-roberta-base")
"""

model = BigBirdPegasusForConditionalGeneration.from_pretrained(
    "google/bigbird-pegasus-large-arxiv")
tokenizer = PegasusTokenizer.from_pretrained(
    "google/bigbird-pegasus-large-arxiv")

text = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."
inputs = tokenizer([text], max_length=4096,
                   return_tensors="pt", truncation=True)

# Generate Summary
summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=20)
print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True,
      clean_up_tokenization_spaces=False))

# select source text for first example
# text = multi_lexsum["train"][0]["description"]
# print(text)
"""
print("Tokenizing...")
inputs = tokenizer(text, return_tensors='pt')
print("Predicting...")
prediction = model.generate(**inputs)

prediction = tokenizer.batch_decode(prediction)
print(len(text))
print(len(prediction[0]))

print(prediction)
"""
