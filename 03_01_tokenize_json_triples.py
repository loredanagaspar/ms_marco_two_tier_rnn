# tokenize_json_triples.py

import os
import re
import pickle
import json
from tqdm import tqdm
from collections import Counter


def simple_tokenize(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"(\w)'(\w)", r"\1_APOS_\2", text)
    for char in '"#$%&()*+,-./:;<=>?@[\\]^_`{|}~':
        text = text.replace(char, ' ')
    text = text.replace('_APOS_', "'")
    return [t for t in text.split() if t]


def preprocess_triples_json(input_file, output_file, vocab_output=None):
    print(f"Tokenizing JSON triples from {input_file}...")

    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Loading...")
        with open(output_file, 'rb') as f:
            return pickle.load(f)

    with open(input_file, 'r', encoding='utf-8') as f:
        triples = json.load(f)

    tokenized_data = {
        'tokenized_queries': [],
        'tokenized_positives': [],
        'tokenized_negatives': []
    }

    vocab_counter = Counter()

    for item in tqdm(triples, desc="Tokenizing"):
        q = simple_tokenize(item['query'])
        p = simple_tokenize(item['pos_doc'])
        n = simple_tokenize(item['neg_doc'])

        tokenized_data['tokenized_queries'].append(q)
        tokenized_data['tokenized_positives'].append(p)
        tokenized_data['tokenized_negatives'].append(n)

        vocab_counter.update(q + p + n)

    print(f"Saving tokenized data to {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(tokenized_data, f)

    if vocab_output:
        print(f"Saving vocabulary to {vocab_output}")
        vocab = sorted(vocab_counter.keys())
        with open(vocab_output, 'wb') as f:
            pickle.dump(vocab, f)

    print("Tokenization complete.")
    print(f"Sample query: {tokenized_data['tokenized_queries'][0]}")
    print(f"Sample positive doc: {tokenized_data['tokenized_positives'][0][:20]}...")
    print(f"Sample negative doc: {tokenized_data['tokenized_negatives'][0][:20]}...")

    return tokenized_data


if __name__ == "__main__":
    preprocess_triples_json(
        "new_triplets/triples_full.json",
        "tokenizedjson/train_tokenized.pkl",
        vocab_output="new_vocab.pkl"
    )
    preprocess_triples_json(
        "new_triplets/triples_test.json",
        "tokenizedjson/test_tokenized.pkl"
    )
    preprocess_triples_json(
        "new_triplets/triples_validation.json",
        "tokenizedjson/validation_tokenized.pkl"
    )
