""" Pretraining on the MLM objective (deprecated) """
import os
import collections
import torch
import numpy as np
from transformers import AutoTokenizer, \
    BigBirdPegasusForConditionalGeneration, \
    BigBirdForMaskedLM, \
    Trainer, \
    DataCollatorForLanguageModeling, \
    default_data_collator, \
    TrainingArguments, Trainer, BigBirdForPreTraining
from datasets import load_dataset
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import evaluate
from evaluate import evaluator


class Pretraining:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Setting up Pretraining on {self.device}")
        self.model = BigBirdForMaskedLM.from_pretrained(
            "google/bigbird-roberta-base")
        # self.model = BigBirdForMaskedLM.from_pretrained(
        #    "google/bigbird-pegasus-large-arxiv")
        # self.migrate_weights_from_pegasus()
        # self.load_billsum_dataset()
        # make sure to pretrain on the Pegasus tokenizer to preserve mapping
        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/bigbird-pegasus-large-arxiv")
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #    "google/bigbird-roberta-base")
        self.load_billsum_dataset()
        self.metric = evaluate.load("rouge")
        self.data_collator = DataCollatorForLanguageModeling(
            self.tokenizer)
        # hyperparameters
        self.BATCH_SIZE = 4  # breaks when we hit 8
        self.N_EPOCHS = 3
        self.CHUNK_SIZE = 512
        self.PROB_MASK = 0.2  # chance of a word being masked out

    def migrate_weights_from_pegasus(self, checkpoint="google/bigbird-pegasus-large-arxiv", name="zeroshot"):
        """ Load in Google's weights from the Pegasus variant """
        print(
            f"Loading in BigBirdPegasus pretrained weights from {checkpoint}")
        pegasus = BigBirdPegasusForConditionalGeneration.from_pretrained(
            checkpoint)
        # dump weight config
        pegasus_path = f"./bigbirdpegasus-{name}"
        if os.path.exists(pegasus_path):
            print("\tWeights already downloaded from BigBirdPegasus")
        else:
            print("\tDownloading weights from BigBirdPegasus")
            trainer = Trainer(pegasus)
            trainer.save_model(pegasus_path)
        # load in weights
        self.model = BigBirdForMaskedLM.from_pretrained(pegasus_path)

    def load_billsum_dataset(self):
        #dataset = load_dataset("billsum")
        self.train = load_dataset("billsum", split="train[:1%]")
        self.test = load_dataset("billsum", split="test[:1%]")

    def billsum_preprocess_function(self, examples):
        """ Tokenization for MLM """
        result = self.tokenizer(examples["text"])
        if self.tokenizer.is_fast:

            result["word_ids"] = [result.word_ids(
                i) for i in range(len(result["input_ids"]))]
        return result

    def group_texts(self, examples):
        """ 
        MLM Pretraining (taken from Hugging Face) 
        https://huggingface.co/course/chapter7/3?fw=tf#preprocessing-the-data
        """
        # Concatenate all texts
        concatenated_examples = {
            k: sum(examples[k], []) for k in examples.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // self.CHUNK_SIZE) * self.CHUNK_SIZE
        # Split by chunks of max_len
        result = {
            k: [t[i: i + self.CHUNK_SIZE]
                for i in range(0, total_length, self.CHUNK_SIZE)]
            for k, t in concatenated_examples.items()
        }
        # Create a new labels column
        result["labels"] = result["input_ids"].copy()
        return result

    def whole_word_masking_data_collator(self, features):
        for feature in features:
            word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, self.PROB_MASK, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = self.tokenizer.mask_token_id
        feature["labels"] = new_labels

        return default_data_collator(features)

    def train_model(self, output_dir):
        print("Initiating training loop")
        # tokenize data
        print(">>> Tokenizing datasets")
        print("\tTrain...")
        tokenized_train = self.train.map(
            self.billsum_preprocess_function,
            batched=True,  # for multithreading acceleration
            remove_columns=["text", "summary", "title"]
        )
        print("\tTest...")
        tokenized_test = self.test.map(
            self.billsum_preprocess_function,
            batched=True,  # for multithreading acceleration
            remove_columns=["text", "summary", "title"]
        )

        # chunk up dataset
        print(">>> Chunking data")
        print("\tTrain...")
        lm_train = tokenized_train.map(self.group_texts, batched=True)
        print("\tTest...")
        lm_test = tokenized_test.map(self.group_texts, batched=True)

        # Show the training loss with every epoch
        logging_steps = len(tokenized_train) // self.BATCH_SIZE
        # model_name = self.checkpoint.split("/")[-1]

        print(self.tokenizer.decode(lm_train[1]["input_ids"]))
        samples = [lm_train[i] for i in range(2)]
        for sample in samples:
            _ = sample.pop("word_ids")

        for chunk in self.data_collator(samples)["input_ids"]:
            print(f"\n'>>> {self.tokenizer.decode(chunk)}'")
        return

        print(">>> Setting up trainer")
        args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
            per_device_train_batch_size=self.BATCH_SIZE,
            per_device_eval_batch_size=self.BATCH_SIZE,
            fp16=(self.device == "cuda"),
            optim="adafactor",
            logging_steps=logging_steps,
            remove_unused_columns=False
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=lm_train,
            eval_dataset=lm_test,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer
        )

        print(">>> Initiating training")

        trainer.train()


if __name__ == "__main__":
    mlm = Pretraining()
    mlm.train_model("mlm-pretrained")
