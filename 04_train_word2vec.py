# file: train_word2vec_combined.py
import os
import json
import logging
import multiprocessing
from collections import Counter
from tqdm import tqdm
from datasets import load_dataset
from gensim.models import Word2Vec
from gensim.models.word2vec import FAST_VERSION

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if FAST_VERSION < 1:
    print("Warning: Gensim's C-optimized code is not enabled. Training may be slow. Ensure a proper build with Cython.")


def load_text8():
    dataset = load_dataset("afmck/text8", split="train")
    return [sentence.split() for sentence in dataset["text"]]


def load_msmarco():
    dataset = load_dataset("microsoft/ms_marco", "v2.1", split="train")
    sentences = []
    for example in tqdm(dataset, desc="Processing MS MARCO"):
        for field in ["query", "passage"]:
            text = example.get(field, "")
            tokens = text.strip().lower().split()
            if tokens:
                sentences.append(tokens)
    return sentences


def analyze_vocab(sentences, min_count):
    counter = Counter()
    for sent in tqdm(sentences, desc="Counting words"):
        counter.update(sent)
    kept = {w for w, c in counter.items() if c >= min_count}
    print(
        f"Total words: {sum(counter.values())}, Unique: {len(counter)}, Kept (min_count={min_count}): {len(kept)}")
    return counter


def save_vocab(counter, path):
    vocab = {word: count for word, count in counter.items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved vocabulary to {path}")


def save_embeddings(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for word in model.wv.index_to_key:
            vec = model.wv[word]
            vec_str = " ".join(map(str, vec))
            f.write(f"{word} {vec_str}\n")
    print(f"Saved embeddings to {path}")


def train_word2vec(sentences, output_path, vector_size, window, min_count, workers, epochs, sg):
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers or multiprocessing.cpu_count(),
        epochs=epochs,
        sg=sg
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    print(f"Model saved to {output_path}, vocab size: {len(model.wv)}")
    return model


def main():
    output_path = "models/word2vec_combined.bin"
    vector_size = 300
    window = 5
    min_count = 5
    epochs = 5
    sg = 1

    print("Loading datasets...")
    sentences = load_text8() + load_msmarco()
    print(f"Loaded {len(sentences)} sentences total")

    vocab_counter = analyze_vocab(sentences, min_count)
    save_vocab(vocab_counter, "models/vocab.json")

    model = train_word2vec(
        sentences,
        output_path=output_path,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=None,
        epochs=epochs,
        sg=sg
    )

    save_embeddings(model, "models/word_vectors.txt")


if __name__ == "__main__":
    main()
