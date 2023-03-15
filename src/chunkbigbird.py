"""
BigBirdPegasus model that implements the custom chunking mechanism
Source: https://stackoverflow.com/questions/70814490/uploading-models-with-custom-forward-functions-to-the-huggingface-model-hub
"""
import tqdm

from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BigBirdPegasusConfig, PreTrainedModel, PretrainedConfig
from evaluate import evaluator

import torch
import itertools

device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CHUNK_SIZE = 4096


class ChunkingBigBird(PreTrainedModel):
    def __init__(self, checkpoint, overlap=100):
        super(ChunkingBigBird, self).__init__(
            config=BigBirdPegasusConfig.from_pretrained(checkpoint))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/bigbird-pegasus-large-arxiv")
        self.load_billsum_dataset()
        self.overlap = overlap

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


if __name__ == "__main__":
    checkpoint = input("Checkpoint (ENTER to use pretrained): ")
    if checkpoint == "":
        checkpoint = "google/bigbird-pegasus-large-arxiv"
    chunkbird = ChunkingBigBird(checkpoint)
    # chunkbird.to(device)
    print(chunkbird.evaluate_model())
