# build_vocab_ms_marco.py

import pandas as pd
import re
import pickle
from tqdm import tqdm


def simple_tokenize(text):
    if pd.isna(text):
        return []
    text = text.lower()
    text = re.sub(r"(\w)'(\w)", r"\1_APOS_\2", text)
    for char in '"!#$%&()*+,-./:;<=>?@[\\]^_`{|}~':
        text = text.replace(char, ' ')
    text = text.replace('_APOS_', "'")
    return [t for t in text.split() if t]


def extract_vocab_from_csv(csv_paths):
    vocab = set()
    for path in csv_paths:
        print(f"Reading {path}...")
        df = pd.read_csv(path)
        for col in ["query", "positive_doc", "negative_doc"]:
            tqdm.pandas(desc=f"Tokenizing {col}")
            df[col].progress_apply(lambda x: vocab.update(simple_tokenize(x)))
    return sorted(vocab)


def save_vocab(vocab, output_path):
    with open(output_path, "wb") as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary of size {len(vocab)} saved to {output_path}")


if __name__ == "__main__":
    csv_files = ["train_triples.csv", "valid_triples.csv", "test_triples.csv"]
    vocab = extract_vocab_from_csv(csv_files)
    save_vocab(vocab, "vocab.pkl")
    print("Vocabulary extraction and saving completed.")
    print(f"Total vocabulary size: {len(vocab)}")
