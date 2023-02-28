""" This program contains functions to load in datasets used for pretraining and fine-tuning """

from datasets import load_dataset
from transformers import AutoTokenizer

def get_lexsum():
    # Download multi_lexsum locally and load it as a Dataset object 
    dataset = load_dataset("allenai/multi_lexsum", name="v20220616")
    dataset = dataset.remove_columns(["summary/tiny", "summary/short"])
    return dataset


if __name__ == "__main__":
    multi_lexsum = get_lexsum()

    print(multi_lexsum['train'][:5] + "\n")

    example = multi_lexsum["validation"][0] # The first instance of the dev set 
    example["sources"] # A list of source document text for the case

    # for sum_len in ["long", "short", "tiny"]:
    #     # # length = len(example["summary/" + sum_len])
    #     # length = 0
    #     # print(f"{sum_len}: {length}")
    #     print(sum_len)
    #     print(example["summary/" + sum_len]) # Summaries of three lengths