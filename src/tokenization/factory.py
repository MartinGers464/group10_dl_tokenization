from .word_tokenizer import WordTokenizer
from .bpe_tokenizer import BPETokenizer
from .unigram_tokenizer import UnigramTokenizer
from .byte_tokenizer import ByteTokenizer

def get_tokenizer(name: str):
    if name == "word":
        return WordTokenizer.load("data/tokenizers/word")
    if name == "bpe":
        return BPETokenizer.load("data/tokenizers/bpe")
    if name == "unigram":
        return UnigramTokenizer.load("data/tokenizers/unigram")
    if name == "byte":
        return ByteTokenizer.load()
    raise ValueError(f"Unknown tokenizer {name}")
