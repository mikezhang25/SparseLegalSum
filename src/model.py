from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, TrainingArguments, Trainer, BigBirdPegasusForConditionalGeneration, DataCollatorForSeq2Seq
from data_util import *
from datasets import DatasetDict, load_dataset
import evaluate
import numpy as np
import torch 

"""
Setting up class boiler plate for later use
"""
class Model(): 
    def __init__(self): 
        self.model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-arxiv")
        self.tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, self.model)

    def fine_tune(self, dataset:DatasetDict):
        
        print(dataset)
        # tokenized_dataset = dataset.map(self.tokenize_function, batched=True)

        small_dataset = DatasetDict(
            train = dataset['train'].shuffle(seed=1111).select(range(48)), 
            val = dataset['test'].shuffle(seed=1111).select(range(32))
        )

        # print(small_dataset["train"])

        #TODO: Remove NONE values from the output of the short summaries

        tokenized_test_set = small_dataset["train"].map(self.preprocess_function)
        tokenized_eval_set = small_dataset["val"].map(self.preprocess_function)


        # print(tokenized_eval_set[0])
        # tokenized_datasets = dataset.map(self.tokenize_function, batched=True)
        # small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(500)
        # small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(500)


        training_args = Seq2SeqTrainingArguments(
            output_dir="my_awesome_seq2seq_model",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=2,
            predict_with_generate=True,
            # fp16=True,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_test_set, 
            eval_dataset=tokenized_eval_set, 
            tokenizer=self.tokenizer,
            # data_collator=self.data_collator,
            # compute_metrics=self.compute_metrics,
        )

        trainer.train()
        
    def lexsum_preprocess_function(self, examples): 
        res = ""
        for doc in examples["sources"]:
            res += doc
        examples["sources"] = [res]

        model_inputs = self.tokenizer(examples["sources"], padding='max_length', truncation=True)
        labels = self.tokenizer(text_target=examples["summary/long"], padding='max_length', truncation=True)
        # print("short exapmles: ", examples["summary/long"])
        model_inputs["labels"] = labels["input_ids"]

        # print(model_inputs["attention_mask"].size())
        # assert False
        # print("Model Inputs: ", model_inputs)
        return model_inputs
    
    def compute_metrics(self, eval_pred):
        rouge = evaluate.load("rouge")
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}
