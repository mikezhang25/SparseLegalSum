""" Trying to train a model, save parameters, then load into a model for other purpose """
import torch
from transformers import AutoTokenizer, BigBirdPegasusForQuestionAnswering, BigBirdPegasusForConditionalGeneration, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
from tqdm import tqdm
import numpy as np
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
import evaluate
from evaluate import evaluator


class LegalModel:
    def __init__(self, checkpoint="google/bigbird-pegasus-large-arxiv") -> None:
        # TODO: Add more modularity to FineTuning Class & Make Legal Model Class
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Setting up finetuning on {self.device}")
        self.model = BigBirdPegasusForConditionalGeneration.from_pretrained(
            checkpoint)
        self.model.to(self.device)
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #    "t5-small")
        # self.tokenizer = AutoTokenizer.from_pretrained()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/bigbird-pegasus-large-arxiv")
        self.load_billsum_dataset()
        self.metric = evaluate.load("rouge")
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, model=self.model)
        # hyperparameters
        self.checkpoint = checkpoint
        self.BATCH_SIZE = 10000
        self.EPOCHS = 8
        self.VOCAB_SIZE = 35000
        self.MAX_LENGTH = 512
        self.MAX_TARGET_SIZE = 512

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
            "billsum", split="train[:1%]")
        self.test = load_dataset(
            "billsum", split="test[:1%]")

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

    def train_model(self):
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
            output_dir="billsum-finetuned",
            evaluation_strategy="epoch",
            learning_rate=5.6e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=num_train_epochs,
            predict_with_generate=True,
            fp16=(self.device == "cuda"),
            optim="adafactor",
            # compute_objective=lambda x: x,
            logging_steps=logging_steps
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
            compute_metrics=self.compute_metrics,
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
    #pretrain = Pretraining()
    #tokens = pretrain.tokenizer("Sample text is true")

    legalModel = LegalModel(checkpoint="google/bigbird-roberta-base")
    # legalModel = LegalModel("billsum-finetuned/checkpoint-9000")
    legalModel.train_model()
    # print(legalModel.evaluate_model())
    # print(legalModel.sample_summary())
