""" Trying to train a model, save parameters, then load into a model for other purpose """
import os
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BigBirdPegasusForConditionalGeneration, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
from tqdm import tqdm
import numpy as np
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
import evaluate
from evaluate import evaluator
import argparse

from createChunks import createChunksfromDataset


class LegalModel:
    def __init__(self, dataset, checkpoint="google/bigbird-pegasus-large-arxiv") -> None:
        # TODO: Add more modularity to FineTuning Class & Make Legal Model Class
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Setting up finetuning on {self.device}")
        # self.model = BigBirdPegasusForConditionalGeneration.from_pretrained(
        #    checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/bigbird-pegasus-large-arxiv")
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #    "nsi319/legal-pegasus")
        # self.load_billsum_dataset()
        # self.load_chunked_dataset()
        if dataset == "non-overlap":
            self.load_overlapping_chunked_dataset()
        elif dataset == "overlap":
            self.load_overlapping_chunked_dataset()
        elif dataset == "summarized":
            self.load_summarized_dataset()
        elif dataset == "billsum":
            self.load_billsum_dataset()
        else:
            raise Exception(
                f"Invalid dataset {dataset} specified during LegalModel initialization")
        self.metric = evaluate.load("rouge")
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, model=self.model)
        wandb.init(project="LegalSparseSum")
        # hyperparameters
        self.checkpoint = checkpoint
        self.BATCH_SIZE = 10000
        self.EPOCHS = 8
        self.VOCAB_SIZE = 35000
        self.MAX_LENGTH = 512
        self.MAX_TARGET_SIZE = 512

    def load_chunked_dataset(self):
        self.train, self.test = createChunksfromDataset("chunked")
        print("Train: ", self.train)
        print("Test: ", self.test)

    def load_overlapping_chunked_dataset(self):
        self.train, self.test = createChunksfromDataset(
            "chunked", k_sentences=3)

    def load_summarized_dataset(self):
        print("Loading in summarized dataset")
        root_filename = input("Root name for summarized document: ")

        # print("Datafiles", data_files)
        self.train = load_dataset(
            "json", data_files=f"{root_filename}_train.json", split="train")
        self.test = load_dataset(
            "json", data_files=f"{root_filename}_test.json", split="train")

    def load_arxiv_dataset(self):
        self.train = load_dataset(
            "ccdv/arxiv-summarization", split="train[:5%]")
        self.test = load_dataset(
            "ccdv/arxiv-summarization", split="validation[:5%]")

    def load_multilex_dataset(self):
        self.train = load_dataset(
            "allenai/multi_lexsum", name="v20220616", split="train[:1%]")
        self.test = load_dataset(
            "allenai/multi_lexsum", name="v20220616", split="validation[:1%]")

    def load_billsum_dataset(self):
        self.train = load_dataset(
            "billsum", split="train")
        self.test = load_dataset(
            "billsum", split="test")

        print(self.train)
        print(self.test)

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
        result = {key: value *
                  100 for key, value in result.items()}
        return {k: round(v, 4) for k, v in result.items()}

    def train_model(self, output_dir):
        # freeze layers before training
        for param in self.model.base_model.parameters():
            # param.requires_grad = False
            param.requires_grad = True
        # tokenize data
        tokenized_train = self.train.map(
            self.billsum_preprocess_function,
            batched=True,  # for multithreading acceleration
        )

        batch_size = 4  # breaks when we hit 8
        num_train_epochs = 3
        # Show the training loss with every epoch
        logging_steps = len(tokenized_train) // batch_size
        # model_name = self.checkpoint.split("/")[-1]
        args = Seq2SeqTrainingArguments(
            report_to="wandb",
            output_dir=output_dir,
            overwrite_output_dir=False,
            evaluation_strategy="steps",
            logging_steps=100,
            eval_steps=500,
            eval_accumulation_steps=1,
            learning_rate=5.6e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=num_train_epochs,
            predict_with_generate=True,
            fp16=(self.device == "cuda"),
            optim="adafactor"
        )

        tokenized_train = tokenized_train.remove_columns(
            self.train.column_names)

        features = [tokenized_train[i] for i in range(2)]

        # print("Here are the features: ",  features)

        self.data_collator(features)

        tokenized_test = self.test.map(
            self.billsum_preprocess_function)

        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        print("Training")
        trainer.train()

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

    def lexsum_preprocess_function(self, examples):
        sources = examples["sources"]
        concat_sources = ""
        for s in sources:
            concat_sources += str(s)
        n = 4096
        chunks = [str[i:i+n] for i in range(0, len(str), n)]

        model_inputs = self.tokenizer(
            str(examples["sources"]), max_length=self.MAX_LENGTH, truncation=True)
        labels = self.tokenizer(
            text_target=examples["summary/long"], max_length=self.MAX_TARGET_SIZE, truncation=True)
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def billsum_preprocess_function(self, examples):
        model_inputs = self.tokenizer(
            examples["text"], max_length=self.MAX_LENGTH, truncation=True)
        labels = self.tokenizer(
            text_target=examples["summary"], max_length=self.MAX_TARGET_SIZE, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def evaluate_model(self):
        """ Returns ROUGE scores for current model """
        data = self.test.shuffle().select(range(100))
        task_evaluator = evaluator("summarization")
        results = task_evaluator.compute(
            model_or_pipeline=self.model, tokenizer=self.tokenizer, data=data, input_column="text", label_column="summary")
        return results

    def sample_summary(self):
        """ Print out the summary of a random sample """
        text = legalModel.test["text"][0]
        input_tokenized = self.tokenizer.encode(
            text, truncation=True, return_tensors='pt')
        input_tokenized = input_tokenized.to(self.device)
        summary_ids = self.model.to(self.device).generate(input_tokenized,
                                                          length_penalty=3.0,
                                                          min_length=30,
                                                          max_length=100)
        output = self.tokenizer.decode(
            summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="train or evaluate a given model")
    parser.add_argument(
        "model_path", help="path to directory to save model file (train model) or saved model file (test mode)")
    parser.add_argument(
        "dataset", help="specifies which dataset we train/test on")
    args = parser.parse_args()

    RUN_NAME = "chunking_finetune"
    # set up logging
    # set the wandb project where this run will be logged
    os.environ["WANDB_PROJECT"] = RUN_NAME
    # save your trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"] = "true"
    # turn off watch to log faster
    os.environ["WANDB_WATCH"] = "false"

    if args.mode == "train":
        # check that path is a folder and is empty
        assert not os.path.isdir(args.model_path) or os.listdir(
            args.model_path) == 0, f"[TRAIN ERROR] {args.model_path} is not an empty directory"
        print(f"Entering train mode, saving model to {args.model_path}")
        legalModel = LegalModel(args.dataset)
        legalModel.train_model(args.model_path)
    elif args.mode == "test":
        # check that checkpoint file exists
        assert os.path.isdir(
            args.model_path), f"[TEST ERROR] checkpoint {args.model_path} does not exist"
        print(f"Entering test mode, loading model from {args.model_path}")
        legalModel = LegalModel(args.dataset, checkpoint=args.model_path)
        results = legalModel.evaluate_model()
        print(results)
    else:
        print(f"Unrecognized mode {args.mode} specified")
    # legalModel = LegalModel("billsum-finetuned/checkpoint-9000")
    # print(legalModel.evaluate_model())
    # print(legalModel.sample_summary())
    # wandb.finish()
