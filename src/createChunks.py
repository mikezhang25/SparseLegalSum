import numpy as np 
import nltk
from datasets import load_dataset
from transformers import AutoTokenizer    
import json
from pathlib import Path
import math

def load_billsum_dataset():
    return load_dataset("billsum")


def _docsToDict(docs, name_of_split, size_of_chunks=4096, appendTitle=False, num_of_chunks=None, k_sentences=None): 
    dataEntries = []
    for doc in docs: 
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
                # dataEntries[name_of_split].append(mapping)
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
            curChunk = curChunk + " " + sentence
            sent_arr.append(sentence)
            
        # concatenate any non-full chunks that remain
        if runningLen > 0: 
            mapping = {"text": curChunk, "summary": doc["summary"]}
            # dataEntries[name_of_split].append(mapping)
            dataEntries.append(mapping)

    # if (name_of_split == "test"):
    #     print("Test Entries: ", dataEntries[name_of_split])
    return dataEntries
    



def chunksToJSON(filename, size_of_chunks=4096, num_of_chunks=None, include_title=False, k_sentences=None, appendTitle=False): 
    dataset = load_billsum_dataset()    
      # where every dictionary entry will be stored
    # default
    
    trainEntries = _docsToDict(dataset["train"], name_of_split="train")
    testEntries = _docsToDict(dataset["test"], name_of_split="test")

    with open (filename + 'train' + '.json', "w") as f: 
        json.dump(trainEntries, f)

    with open (filename + 'test' + '.json', "w") as f: 
        json.dump(testEntries, f)

def JSONToDataset(filename, name_of_split): 
    return load_dataset("json", data_files=filename, split=name_of_split)

def createChunksfromDataset(filename, size_of_chunks=4096, num_of_chunks=None, recreate=False, k_sentences=None, appendTitle=False): 
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
    testName = filename + 'test' + '.json'
    trainName = filename + 'train' + '.json'
    testDataset = JSONToDataset(testName, "train")
    trainDataset = JSONToDataset(trainName, "train")
    return trainDataset, testDataset


def createChunksFromDoc(doc, num_of_chunks, k_sentences=None): 
    """
    Breaks down an input string document into
    the number of chunks specified. 

    Args:
        doc (string): text document
        num_of_chunks (int):  number of chunks desired
        k_sentences (int, optional): Number of overlapping chunks desired. Defaults to None.

    Returns:
        chunks: list of strings
    """
    chunks = []
    size_of_chunks = math.ceil(len(doc)/num_of_chunks) # applies to this specific document
    sent_text = nltk.sent_tokenize(doc)
    runningLen = 0
    curChunk = ""
    sent_arr = []
    for sentence in sent_text: 
        sentLen = len(sentence)

        condition = True if runningLen > size_of_chunks else False

        if condition: 
            # already reached the maximum size for this chunk
            # mapping = {"text": curChunk, "summary": doc["summary"]}
            # dataEntries.append(mapping)
            chunks.append(curChunk)
            runningLen = 0
            curChunk = ""

            # take the last k sentences from this chunk and append to next chunk
            if k_sentences is not None: 
                assert k_sentences >= 1
                last_k_sentences = sent_arr[-k_sentences:] 
                for s in last_k_sentences: 
                    runningLen += len(s)
                    curChunk = " " + s
                sent_arr = []
        
        # add sentence 
        runningLen += sentLen
        curChunk = curChunk + " " + sentence
        sent_arr.append(sentence)
        
    # concatenate any non-full chunks that remain
    if runningLen > 0: 
        chunks.append(curChunk)

    return chunks
     

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


if __name__ == "__main__": 
    filename = "test" # intermediate JSON file
    size_of_chunks = 4096
    num_of_chunks = 3
    k_sentences = 3
    include_title = True

    # actual dataset
    train, test = createChunksfromDataset(filename, 
                            recreate=True,
                            )
    print("Test: ", test)
    print("Train: ", train)


    '''
    The following implementation is for one document. It returns a list of strings. 
    The size of the list is <= num_of_chunks. 
    '''

    # for testing: allows me to see a few entries at a time
    # with open ("test.json", 'w') as f: 
    #     json.dump(dataset[0:5], f, indent=4)

    # doc = "An alternative to solving for the optimal value function over the belief space is to use posterior sampling,8 which was originally introduced in the context of exploration in bandit problems in section 15.4.9 Here, we draw a sample θ from the current belief b and then solve for the best action, assuming that θ is the true model. We then update our belief, draw a new sample, and solve the corresponding MDP. Example 16.4 provides an example instance of this. An advantage of posterior sampling is that we do not have to decide on heuristic exploration parameters. However, solving the MDP at every step can be expensive. A method for sampling a discrete MDP from the posterior is implemented in algorithm 16.9."

    # chunks = createChunksFromDoc(doc, 5)

    # print("Length of Chunks: ", len(chunks))
    # print("Chunks: ", chunks)