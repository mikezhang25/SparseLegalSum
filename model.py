from transformers import BigBirdForPreTraining, AutoTokenizer

def configure_model():
    """ 
    Loads in the Big Bird model from pretrained weights
    Returns the model and its corresponding tokenizer
    """
    model = BigBirdForPreTraining.from_pretrained("google/bigbird-roberta-base")
    tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
    return model, tokenizer