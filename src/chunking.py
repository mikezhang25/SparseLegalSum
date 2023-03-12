""" 
Program to chunk up documents in a database and construct a new dataset from the concatenate summaries
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize
from evaluate import evaluator
import torch
from tqdm import tqdm
from datasets import load_dataset
import json


class Summarizer:
    def __init__(self) -> None:
        # https://huggingface.co/nsi319/legal-pegasus
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Summarizer on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained("nsi319/legal-pegasus")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "nsi319/legal-pegasus")
        self.model.to(self.device)
        # hyperparameters
        self.N_BEAMS = 9
        self.NO_REPEAT_NGRAM = 3
        self.PENALTY = 5.0
        self.MAX_OUTPUT_LEN = 4096

    def summarize(self, tokenized, min_length, max_length):
        """ Summarizes a given **tokenized** chunk of text to a target length """
        tokenized.to(self.device)
        summary_ids = self.model.generate(tokenized,
                                          num_beams=self.N_BEAMS,
                                          no_repeat_ngram_size=self.NO_REPEAT_NGRAM,
                                          length_penalty=self.PENALTY,
                                          min_length=min_length,
                                          max_length=max_length,
                                          early_stopping=True)
        summary = [self.tokenizer.decode(
            g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
        return summary

    def chunk(self, document, chunk_size=1024):
        """ Splits up a document into chunks while perserving sentence structure """
        sent_encodes = [self.tokenizer.encode(
            sent, max_length=1024, truncation=True) for sent in sent_tokenize(document)]
        max_sent_len = max([len(x) for x in sent_encodes])
        if max_sent_len > chunk_size:
            raise Exception(
                f"Cannot chunk into sizes smaller than a single sentence. Max sentence is length {max_sent_len}")
        chunks = []
        curr_chunk = []
        for sentence in tqdm(sent_encodes, leave=False, desc=f"Chunking document of {len(sent_encodes)} sentences"):
            # print(
            #    f"Current chunk of size {len(curr_chunk)}, {curr_chunk}, current sentence of size {len(sentence)}")
            if len(curr_chunk) + len(sentence) > chunk_size:
                assert(len(curr_chunk) > 0)
                # declare new chunk
                chunks.append(torch.tensor(curr_chunk).reshape(
                    (1, -1)).to(self.device))
                curr_chunk = []
            # append to existing chunk
            curr_chunk += sentence
        if len(curr_chunk) > 0:
            chunks.append(torch.tensor(
                curr_chunk).reshape(1, -1).to(self.device))
        # assert(sum([len(x) for x in sent_encodes])
        #       == sum([len(x) for x in chunks]))
        #print([len(x) for x in sent_encodes])
        return chunks

    def process(self, document, min_chunk_size=100, chunk_size=1024):
        """ Given document reduces its size via chunking summarization """
        chunks = self.chunk(document, chunk_size=chunk_size)
        target_len = self.MAX_OUTPUT_LEN // len(chunks)
        target_len = max(min_chunk_size, target_len)
        output = ""
        for chunk in tqdm(chunks, leave=False, desc=f"Summarizing document of {len(chunks)} chunks"):
            output += self.summarize(chunk, 0, target_len)
        return output

    def transform_dataset(self, filename):
        billsum = {
            "train": load_dataset("billsum", split="train[:10]"),
            "test": load_dataset("billsum", split="test[:10]"),
            "ca_test": load_dataset("billsum", split="ca_test[:10]")
        }
        summarized = {}
        for split in ["train", "test", "ca_test"]:
            dataset = billsum[split]
            summarized[split] = []
            for data in tqdm(dataset, leave=False, desc=f"Mapping {split} set..."):
                summarized[split].append({
                    "text": self.process(data["text"]),
                    "summary": data["summary"],
                    "title": data["title"]
                })
            # make this into a list of dictionaries then json dump
            print(f"Savung {filename}_{split}.json")
            with open(f"{filename}_{split}.json", "w") as f:
                json.dump(summarized[split], f)


if __name__ == "__main__":
    text = """On March 5, 2021, the Securities and Exchange Commission charged AT&T, Inc. with repeatedly violating Regulation FD, and three of its Investor Relations executives with aiding and abetting AT&T's violations, by selectively disclosing material nonpublic information to research analysts. According to the SEC's complaint, AT&T learned in March 2016 that a steeper-than-expected decline in its first quarter smartphone sales would cause AT&T's revenue to fall short of analysts' estimates for the quarter. The complaint alleges that to avoid falling short of the consensus revenue estimate for the third consecutive quarter, AT&T Investor Relations executives Christopher Womack, Michael Black, and Kent Evans made private, one-on-one phone calls to analysts at approximately 20 separate firms. On these calls, the AT&T executives allegedly disclosed AT&T's internal smartphone sales data and the impact of that data on internal revenue metrics, despite the fact that internal documents specifically informed Investor Relations personnel that AT&T's revenue and sales of smartphones were types of information generally considered "material" to AT&T investors, and therefore prohibited from selective disclosure under Regulation FD. The complaint further alleges that as a result of what they were told on these calls, the analysts substantially reduced their revenue forecasts, leading to the overall consensus revenue estimate falling to just below the level that AT&T ultimately reported to the public on April 26, 2016. The SEC's complaint, filed in federal district court in Manhattan, charges AT&T with violations of the disclosure provisions of Section 13(a) of the Securities Exchange Act of 1934 and Regulation FD thereunder, and charges Womack, Evans and Black with aiding and abetting these violations. The complaint seeks permanent injunctive relief and civil monetary penalties against each defendant. The SEC's investigation was conducted by George N. Stepaniuk, Thomas Peirce, and David Zetlin-Jones of the SEC's New York Regional Office. The SEC's litigation will be conducted by Alexander M. Vasilescu, Victor Suthammanont, and Mr. Zetlin-Jones. The case is being supervised by Sanjay Wadhwa."""
    summarizer = Summarizer()
    filename = input("Base filename: ")
    summarizer.transform_dataset(filename)
    # sent_tokens = summarizer.chunk(text, chunk_size=150)
    # print(sent_tokens[0])
    """
    input = summarizer.tokenizer(text)
    summarizer.model.generate(torch.tensor(input["input_ids"]))
    summary = summarizer.tokenizer.decode(
        text, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(summary)
    """
    #input_tokenized = summarizer.tokenizer.encode(text, return_tensors='pt',max_length=1024,truncation=True)
    """
    chunks = summarizer.chunk(text, chunk_size=100)
    print([x.shape for x in chunks])
    summary_ids = summarizer.model.generate(chunks[0],
                                            num_beams=9,
                                            no_repeat_ngram_size=3,
                                            length_penalty=2.0,
                                            min_length=150,
                                            max_length=250,
                                            early_stopping=True)
    summary = [summarizer.tokenizer.decode(
        g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
    """
    # results = summarizer.process(text, chunk_size=100)
    summarizer.transform_dataset("summarized")

    # decoded = [summarizer.tokenizer.decode(
    #    text, skip_special_tokens=True, clean_up_tokenization_spaces=False) for text in sent_tokens]
