from transformers import BigBirdForPreTraining, AutoTokenizer, TrainingArguments, Trainer
from data_util import *
from datasets import DatasetDict


LOCAL = True
"""
Setting up class boiler plate for later use
"""
class Model(): 
    def __init__(self): 
        self.model = BigBirdForPreTraining.from_pretrained("google/bigbird-roberta-base")
        self.tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")

    def fine_tune(self, dataset):

        # tokenized_dataset = dataset.map(self.tokenize_function, batched=True)
        
        small_dataset = DatasetDict(
            train = dataset['train'].shuffle(seed=1111).select(range(128)).map(self.tokenize_function), 
            val = dataset['test'].shuffle(seed=1111).select(range(32)).map(self.tokenize_function), 
        )

        training_args = TrainingArguments(
            output_dir="trainer_dir",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            evaluation_strategy="epoch", # run validation at the end of each epoch
            save_strategy="epoch",
            learning_rate=2e-5,
            load_best_model_at_end=True,
            seed=224
        )
        trainer = Trainer(
            model=self.model, 
            args=training_args,
            train_dataset=small_dataset['train'], 
            eval_dataset=small_dataset['eval'], 
            tokenizer = self.tokenizer, 
            # compute_metrics=compute_metrics
        )

        trainer.train()

    def tokenize_function(self, examples):
        if LOCAL: 
            return self.tokenizer(examples["text"], padding="max_length", truncation=True)
        else: 
            return self.tokenizer(examples["text"], padding="max_length", truncation=True)
        

