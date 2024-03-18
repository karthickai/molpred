from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split
from tokenizers import Regex
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from transformers import DebertaTokenizerFast


def bpe_tokenizer(path="./data/pretrain.txt", save_to="./data/bpe/"):
    # Initialize the tokenizer with the BPE model and unknown token placeholder
	tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # Set pre-tokenizer to split the input on brackets, which are removed from the tokens
	tokenizer.pre_tokenizer = Split(pattern=Regex("\[|\]"), behavior="removed")
    
    # Configure post-processor to add special tokens [CLS] and [SEP] around the input
	tokenizer.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1", special_tokens=[("[CLS]", 1), ("[SEP]", 2)],)


    # Define a trainer with special tokens needed for the BPE tokenizer
	trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    
    # Train the tokenizer on the specified file(s)
	tokenizer.train(files=[path], trainer=trainer)
    
    # Save the trained model
	tokenizer.save(save_to + "/bpe.json", pretty=True)
	tokenizer.model.save(save_to)

bpe_tokenizer()
# Convert the BPE into DeBERTa Fast Tokenizer
tokenizer = DebertaTokenizerFast.from_pretrained("./data/bpe")
tokenizer.save_pretrained("./data/debertaTokenizer")
