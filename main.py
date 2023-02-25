from model import *
from transformers import AutoTokenizer, BigBirdForPreTraining
import torch

if __name__ == "__main__":
    # model, tokenizer = configure_model()
    tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
    model = BigBirdForPreTraining.from_pretrained(
        "google/bigbird-roberta-base")

    inputs = tokenizer("Hugging Face Transformers is great!",
                       return_tensors="pt", truncation=True, padding = True)
    
    outputs = model(**inputs)

    prediction_logits = outputs.prediction_logits
    seq_relationship_logits = outputs.seq_relationship_logits

    print(tokenizer.batch_decode(inputs.input_ids, skip_special_tokens=True))
