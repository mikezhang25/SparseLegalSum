from model import *
from data_util import *
from transformers import AutoTokenizer, BigBirdForPreTraining
import torch
import data_util

if __name__ == "__main__":
    model = Model()
    model.fine_tune(get_lexsum())
    # dataset = get_lexsum()
    # print(dataset["train"]["sources"][0][0])
