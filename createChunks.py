import numpy as np 
import nltk
from datasets import load_dataset
from transformers import AutoTokenizer    
import json
from pathlib import Path
import math

def load_billsum_dataset():
    return load_dataset("billsum")


def discreteChunksToJSON(filename, size_of_chunks=4096, num_of_chunks=None, include_title=False): 
    dataset = load_billsum_dataset()
    i = 0
    tokenizer = AutoTokenizer.from_pretrained(
            "google/bigbird-pegasus-large-arxiv")
    
    dataEntries = []  # where every dictionary entry will be stored
    # default

    for doc in dataset["train"]: 
        if num_of_chunks != None: 
            size_of_chunks = math.ceil(len(doc["text"])/num_of_chunks) # applies to this specific document
        sent_text = nltk.sent_tokenize(doc["text"])
        runningLen = 0
        curChunk = ""
        for sentence in sent_text: 
            sentLen = len(sentence)

            if size_of_chunks != None: 
                condition = True if runningLen > size_of_chunks else False
            else: 
                condition = True if runningLen + sentLen > size_of_chunks else False

            if condition: 
                # already reached the maximum size for this chunk
                mapping = {"text": curChunk, "summary": doc["summary"]}
                dataEntries.append(mapping)
                runningLen = 0
                curChunk = ""  

            runningLen += sentLen
            curChunk += sentence
        # concatenate any non-full chunks that remain
        if runningLen > 0: 
            mapping = {"text": curChunk, "summary": doc["summary"]}
            dataEntries.append(mapping)

    with open (filename, "w") as f: 
        json.dump(dataEntries, f)



def JSONToDataset(filename): 
    return load_dataset("json", data_files=filename, split="train")

def createDiscreteChunks(filename, size_of_chunks=4096, num_of_chunks=None, recreate=False): 
    """
    Creates discrete chunks of a given dataset (non-overlapping)

    Args:
        size_of_chunks (int, optional): Defaults to 4096.
        num_of_chunks (_type_, optional):  Defaults to None.
        recreate (bool, optional): Defaults to False.

    Returns:
       Dataset: returns a Dataset object from transformers library
    """
    path = Path("./" + filename)
    if not path.is_file() or recreate: 
    # creates the JSON file with the given filename
        discreteChunksToJSON(filename, 
                            size_of_chunks=size_of_chunks, 
                            num_of_chunks=num_of_chunks
                            ) 
        print("Should Print")
    # json file exists at this point
    newDataset = JSONToDataset(filename)
    return newDataset

def createOverlappingChunks(filename, size_of_chunks=4096, num_of_chunks=None, recreate=False):
    pass
    # TODO: Implement overlapping with K tokens - may have to be K sentences. 

# example of how to use: 

filename = "chunkedDataset.json" # intermediate JSON file
size_of_chunks = 4096
num_of_chunks = 3

# actual dataset
dataset = createDiscreteChunks(filename, num_of_chunks=3, recreate=True)

'''
Please use recreate=True when changing any of the parameters. This ensures that that the new dataset is actually
created. 
Alternatively just create a new intermediate filename and that will create a new dataset with the specified params.
'''

# let's see an entry with 3 chunks
with open ("test.json", 'w') as f: 
    json.dump(dataset[0:3], f, indent=4)