from .bpe_tokenizer import BPETokenizer

def main():
    corpus = "data/tokenizers/corpus.txt"
    BPETokenizer.train(corpus_path=corpus, vocab_size=5000)

if __name__ == "__main__":
    main()
