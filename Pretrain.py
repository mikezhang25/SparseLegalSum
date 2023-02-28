


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
