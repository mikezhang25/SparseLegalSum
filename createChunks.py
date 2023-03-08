import numpy as np 
import nltk
from datasets import load_dataset
from transformers import AutoTokenizer    
import json
from pathlib import Path
import math

def load_billsum_dataset():
    return load_dataset("billsum")


def chunksToJSON(filename, size_of_chunks=4096, num_of_chunks=None, include_title=False, k_sentences=None, appendTitle=False): 
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
        runningLen = 0 if appendTitle == False else len(doc["title"])
        curChunk = "" if appendTitle == False else doc["title"] + " "
        sent_arr = []
        for sentence in sent_text: 
            sentLen = len(sentence)

            if num_of_chunks != None: 
                condition = True if runningLen > size_of_chunks else False
            else: 
                condition = True if runningLen + sentLen > size_of_chunks else False

            if condition: 
                # already reached the maximum size for this chunk
                mapping = {"text": curChunk, "summary": doc["summary"]}
                dataEntries.append(mapping)
                runningLen = 0
                curChunk = ""

                # take the last k sentences from this chunk and append to next chunk
                if k_sentences is not None: 
                    assert k_sentences >= 1
                    last_k_sentences = sent_arr[-k_sentences:] 
                    for s in last_k_sentences: 
                        runningLen += len(s)
                        curChunk += s
                    sent_arr = []
            
            # add sentence 
            runningLen += sentLen
            curChunk += sentence
            sent_arr.append(sentence)
            
        # concatenate any non-full chunks that remain
        if runningLen > 0: 
            mapping = {"text": curChunk, "summary": doc["summary"]}
            dataEntries.append(mapping)

    with open (filename, "w") as f: 
        json.dump(dataEntries, f)



def JSONToDataset(filename): 
    return load_dataset("json", data_files=filename, split="train")

def createChunks(filename, size_of_chunks=4096, num_of_chunks=None, recreate=False, k_sentences=None, appendTitle=False): 
    """
    Creates discrete chunks of a given dataset (non-overlapping)

    Args:
        size_of_chunks (int, optional): Defaults to 4096.
        num_of_chunks (_type_, optional):  Defaults to None.
        recreate (bool, optional): Defaults to False.
        appendTitle (bool, optional): Appends title to the front of the first chunk. 
        k_sentences (int, optional): Number of sentences to keep from previous chunk in next chunk. 
    Returns:
       Dataset: returns a Dataset object from transformers library
    """
    path = Path("./" + filename)
    if not path.is_file() or recreate: 
    # creates the JSON file with the given filename
        chunksToJSON(filename, 
                            size_of_chunks=size_of_chunks, 
                            num_of_chunks=num_of_chunks, 
                            k_sentences=k_sentences, 
                            appendTitle=appendTitle
                            ) 
        
    # json file exists at this point
    newDataset = JSONToDataset(filename)
    return newDataset



'''
There are a few key ways of using the function 
1. Don't pass in anything except the filename
    * This will automatically create chunks of 4096 length
    * Alternatively set the size_of_chunks to your target chunk size

2. Pass in num_of_chunks equal to desired chunk count

3. Pass in k_sentences equal to number of overlapping sentences between chunks

4. You can set appendTitle=True so that the title is included in the first chunk
NOTE: LMK if you think appending the title to each chunk is a good idea. 

Please use recreate=True whenever you modify one of the input variables to the
function. This ensures that the dataset is actually recreated with the modifications. 

Things to avoid: 

1. Don't set k_sentences and num_of_chunks at the same time. 
This will lead to untested behavior. 
2. Don't set num_of_chunks and size_of_chunks at the same time. 
This will lead to untested behavior. 

'''

filename = "chunkedDataset.json" # intermediate JSON file
size_of_chunks = 4096
num_of_chunks = 3
k_sentences = 3
include_title = True


# actual dataset
dataset = createChunks(filename, 
                        recreate=True,
                        k_sentences=k_sentences, 
                        appendTitle=include_title
                        )


# for testing: allows me to see a few entries at a time
with open ("test.json", 'w') as f: 
    json.dump(dataset[0:5], f, indent=4)