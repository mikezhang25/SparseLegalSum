""" Trying to train a model, save parameters, then load into a model for other purpose """
import torch
from transformers import AutoTokenizer, BigBirdPegasusForQuestionAnswering, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
from tqdm import tqdm
import numpy as np
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
import evaluate


class Pretraining:
    def __init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Setting up Pretraining on {device}")
        self.model = BigBirdPegasusForQuestionAnswering.from_pretrained(
            "google/bigbird-pegasus-large-arxiv")
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #    "google/bigbird-pegasus-large-arxiv")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "multilex")
        self.load_dataset()
        # hyperparameters
        self.BATCH_SIZE = 10000
        self.VOCAB_SIZE = 35000

    def load_dataset(self) -> None:
        self.train_dataset = load_dataset(
            "allenai/multi_lexsum", name="v20220616", split="train")
        # only train on the short (1 paragraph) summaries for now
        self.train_dataset = self.train_dataset.remove_columns(
            ["id", "summary/long", "summary/tiny"])
        print(self.train_dataset.column_names)
        # wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])  # only keep the 'text' column

    def batch_iterator(self):
        """ generator function to dynamically load data by batch """
        for i in tqdm(range(0, len(self.train_dataset))):
            yield self.train_dataset[i]["sources"]

    def train_tokenizer(self, save_path):
        """ Train tokenizer to recognize terms in new dataset """
        trained_tokenizer = self.tokenizer.train_new_from_iterator(
            text_iterator=self.batch_iterator(),
            vocab_size=self.VOCAB_SIZE)
        trained_tokenizer.save_pretrained(save_path)


class FineTuning:
    def __init__(self, checkpoint) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Setting up finetuning on {self.device}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "t5-small")
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "t5-small")
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #    "multilex")
        self.load_dataset()
        self.metric = evaluate.load("rouge")
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, model=self.model)
        # hyperparameters
        self.checkpoint = checkpoint
        self.BATCH_SIZE = 10000
        self.EPOCHS = 8
        self.VOCAB_SIZE = 35000
        self.MAX_LENGTH = 512
        self.MAX_TARGET_SIZE = 30

    def load_dataset(self):
        self.train = load_dataset(
            "ccdv/arxiv-summarization", split="train[:1%]")
        self.test = load_dataset(
            "ccdv/arxiv-summarization", split="validation[:1%]")

    def preprocess_function(self, examples):
        model_inputs = self.tokenizer(
            examples["article"],
            max_length=self.MAX_LENGTH,
            truncation=True
        )
        labels = self.tokenizer(
            examples["abstract"], max_length=self.MAX_TARGET_SIZE, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        # Decode generated summaries into text
        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        # Decode reference summaries into text
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True)
        # ROUGE expects a newline after each sentence
        decoded_preds = ["\n".join(sent_tokenize(pred.strip()))
                         for pred in decoded_preds]
        decoded_labels = ["\n".join(sent_tokenize(label.strip()))
                          for label in decoded_labels]
        # Compute ROUGE scores
        result = self.metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        # Extract the median scores
        result = {key: value.mid.fmeasure *
                  100 for key, value in result.items()}
        return {k: round(v, 4) for k, v in result.items()}

    def train_model(self):
        # tokenize data
        tokenized_train = self.train.map(
            self.preprocess_function, batched=True)

        print(len(tokenized_train))

        batch_size = 100
        num_train_epochs = 8
        # Show the training loss with every epoch
        logging_steps = len(tokenized_train) // batch_size
        model_name = self.checkpoint.split("/")[-1]
        args = Seq2SeqTrainingArguments(
            output_dir=f"{model_name}-finetuned-amazon-en-es",
            evaluation_strategy="epoch",
            learning_rate=5.6e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=num_train_epochs,
            predict_with_generate=True,
            fp16=(self.device == "cuda"),
            optim="adafactor"
            logging_steps=logging_steps
        )

        tokenized_train = tokenized_train.remove_columns(
            self.train.column_names)
        features = [tokenized_train[i] for i in range(2)]
        self.data_collator(features)

        tokenized_test = self.test.map(
            self.preprocess_function, batched=True)

        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()


if __name__ == "__main__":
    #pretrain = Pretraining()
    #tokens = pretrain.tokenizer("Sample text is true")
    finetune = FineTuning("arxiv_sum")
    finetune.train_model()
