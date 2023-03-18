"""
BigBirdPegasus model that implements the custom chunking mechanism
Source: https://stackoverflow.com/questions/70814490/uploading-models-with-custom-forward-functions-to-the-huggingface-model-hub
"""
import wandb

from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BigBirdPegasusConfig, PreTrainedModel, PretrainedConfig
from evaluate import evaluator

import torch
import itertools
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CHUNK_SIZE = 4096

wandb.init(project="LegalSparseSum")


class ChunkingBigBird(PreTrainedModel):
    def __init__(self, checkpoint, overlap):
        print(
            f"Loading model from {checkpoint}, initializing with {overlap}-token overlap")
        super(ChunkingBigBird, self).__init__(
            config=BigBirdPegasusConfig.from_pretrained(checkpoint))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/bigbird-pegasus-large-arxiv")
        self.load_billsum_dataset()
        self.overlap = overlap
        # hyperparameters
        self.N_BEAMS = 9
        self.NO_REPEAT_NGRAM = 3
        self.PENALTY = 5.0
        self.MAX_OUTPUT_LEN = 4096

    def load_billsum_dataset(self):
        self.train = load_dataset(
            "billsum", split="train")
        self.test = load_dataset(
            "billsum", split="test")

    def forward(self, sent_id, mask):
        """ Process input texts by chunks and returns a concatenated summary """
        chunks = [sent_id[i * MAX_CHUNK_SIZE - (self.overlap * i): (i + 1) * MAX_CHUNK_SIZE - (self.overlap * i)] for i in range
                  (1 + (len(sent_id) - MAX_CHUNK_SIZE + (MAX_CHUNK_SIZE - self.overlap) - 1) // (MAX_CHUNK_SIZE - self.overlap))]
        chunk_masks = [mask[i * MAX_CHUNK_SIZE - (self.overlap * i): (i + 1) * MAX_CHUNK_SIZE - (self.overlap * i)] for i in range
                       (1 + (len(mask) - MAX_CHUNK_SIZE + (MAX_CHUNK_SIZE - self.overlap) - 1) // (MAX_CHUNK_SIZE - self.overlap))]
        # concat the outputs into one
        results = [self.model(chunks[i], mask=chunk_masks[i])
                   for i in range(len(chunks))]

        return list(itertools.chain.from_iterable(results))

    def evaluate_model(self):
        """ Returns ROUGE scores for current model """
        data = self.test.shuffle().select(range(1000))
        task_evaluator = evaluator("summarization")
        results = task_evaluator.compute(
            model_or_pipeline=self.model, tokenizer=self.tokenizer, data=data, input_column="text", label_column="summary")
        return results

    def summarize(self, text, min_length, max_length):
        """ Summarizes a given chunk of text to a target length """
        tokenized = self.tokenizer(
            text, max_length=4096, truncation=True, return_tensors="pt")
        tokenized.to(self.device)
        summary_ids = self.model.generate(**tokenized,
                                          num_beams=self.N_BEAMS,
                                          no_repeat_ngram_size=self.NO_REPEAT_NGRAM,
                                          length_penalty=self.PENALTY,
                                          min_length=min_length,
                                          max_length=max_length,
                                          early_stopping=True)
        summary = [self.tokenizer.decode(
            g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("overlap", type=int)
    parser.add_argument(
        "--checkpoint", nargs="?", default="google/bigbird-pegasus-large-arxiv", const="google/bigbird-pegasus-large-arxiv")
    args = parser.parse_args()
    print(args.checkpoint)

    chunkbird = ChunkingBigBird(args.checkpoint, args.overlap)
    # chunkbird.to(device)
    # print(chunkbird.evaluate_model())
    summary = chunkbird.summarize(
        chunkbird.test.select(range(1))["text"], 0, 200)
    print(summary)
