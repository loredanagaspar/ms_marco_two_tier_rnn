import pandas as pd
import os
import re
import pickle
import argparse
from tqdm import tqdm


def simple_tokenize(text):
    """
    Tokenize text using a classical approach with simple rules.

    Args:
        text: Input text string

    Returns:
        List of tokens
    """
    # Handle None or NaN values
    if pd.isna(text) or text is None:
        return []

    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Replace punctuation with spaces
    # First save apostrophes in words like "don't" by temporarily replacing them
    text = re.sub(r"(\w)'(\w)", r"\1_APOS_\2", text)

    # Replace all other punctuation with spaces
    for char in '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~':
        text = text.replace(char, ' ')

    # Restore apostrophes
    text = text.replace('_APOS_', "'")

    # Split by whitespace and filter out empty tokens
    tokens = [token for token in text.split() if token]

    return tokens


def preprocess_triples(input_file, output_file):
    """
    Tokenize triples and save the result.

    Args:
        input_file: Path to CSV file containing triples
        output_file: Path to save tokenized data
    """
    print(f"Tokenizing triples from {input_file}...")

    # Check if output file already exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Loading...")
        with open(output_file, 'rb') as f:
            return pickle.load(f)

    # Load triples data
    triples_df = pd.read_csv(input_file)
    print(f"Loaded {len(triples_df)} triples")

    # Container for tokenized data
    tokenized_data = {
        'query_ids': triples_df['query_id'].tolist(),
        'tokenized_queries': [],
        'tokenized_positives': [],
        'tokenized_negatives': []
    }

    # Tokenize all text
    for i, row in tqdm(triples_df.iterrows(), total=len(triples_df), desc="Tokenizing"):
        tokenized_data['tokenized_queries'].append(
            simple_tokenize(row['query']))
        tokenized_data['tokenized_positives'].append(
            simple_tokenize(row['positive_doc']))
        tokenized_data['tokenized_negatives'].append(
            simple_tokenize(row['negative_doc']))

    # Save preprocessed data
    print(f"Saving tokenized data to {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(tokenized_data, f)

    # Print statistics
    total_tokens = (
        sum(len(q) for q in tokenized_data['tokenized_queries']) +
        sum(len(p) for p in tokenized_data['tokenized_positives']) +
        sum(len(n) for n in tokenized_data['tokenized_negatives'])
    )

    avg_query_len = sum(len(
        q) for q in tokenized_data['tokenized_queries']) / len(tokenized_data['tokenized_queries'])
    avg_pos_len = sum(len(p) for p in tokenized_data['tokenized_positives']) / len(
        tokenized_data['tokenized_positives'])
    avg_neg_len = sum(len(n) for n in tokenized_data['tokenized_negatives']) / len(
        tokenized_data['tokenized_negatives'])

    print(f"Total tokens: {total_tokens}")
    print(
        f"Average lengths: query={avg_query_len:.1f}, positive={avg_pos_len:.1f}, negative={avg_neg_len:.1f}")

    # Print a sample
    if tokenized_data['tokenized_queries']:
        print("\nSample tokenization for first triple:")
        print(f"Query: {tokenized_data['tokenized_queries'][0]}")
        print(
            f"Positive doc: {tokenized_data['tokenized_positives'][0][:20]}...")
        print(
            f"Negative doc: {tokenized_data['tokenized_negatives'][0][:20]}...")

    return tokenized_data


def process_all_datasets():
    """Process all datasets and tokenize them."""
    # Create output directory
    os.makedirs("tokenized", exist_ok=True)

    # Input files
    input_files = {
        'train': 'train_triples.csv',
        'valid': 'valid_triples.csv',
        'test': 'test_triples.csv'
    }

    # Process each dataset
    for split, input_file in input_files.items():
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found, skipping")
            continue

        output_file = f"tokenized/{split}_tokenized.pkl"

        print(f"\nProcessing {split} dataset...")
        preprocess_triples(input_file, output_file)

    print("\nAll datasets tokenized!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize triples data")
    parser.add_argument("--input", type=str, help="Input triples CSV file")
    parser.add_argument("--output", type=str,
                        help="Output tokenized data file (.pkl)")
    parser.add_argument("--all", action="store_true",
                        help="Process all datasets")

    args = parser.parse_args()

    if args.all:
        process_all_datasets()
    elif args.input and args.output:
        preprocess_triples(args.input, args.output)
    else:
        parser.print_help()
