
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import random
import ast
import re
import pandas as pd


def validate_and_filter_triples(triples):
    """
    Validate and filter triples to ensure:
    1. No duplicate triples
    2. No cases where positive == negative
    """
    unique_triples = []
    seen = set()
    filtered_count = 0

    for triple in triples:
        # Ensure positive != negative
        if triple['positive_doc'] == triple['negative_doc']:
            filtered_count += 1
            continue

        # Create a key for uniqueness check
        key = (triple['query_id'], triple['positive_doc'],
               triple['negative_doc'])

        # Check if we've seen this triple before
        if key in seen:
            filtered_count += 1
            continue

        # Add to unique triples and track key
        seen.add(key)
        unique_triples.append(triple)

    print(f"Filtered {filtered_count} invalid or duplicate triples")
    print(f"Returning {len(unique_triples)} valid unique triples")

    return unique_triples


def extract_arrays_from_string(passage_str):
    """
    Extract is_selected and passage_text arrays from the specific format
    in the MS Marco dataset.
    """
    try:
        # Use regex to find the arrays
        is_selected_pattern = r"'is_selected':\s*array\(\[(.*?)\]"
        passage_text_pattern = r"'passage_text':\s*array\(\[(.*?)\],\s*dtype=object\)"

        # Extract is_selected
        is_selected_match = re.search(is_selected_pattern, passage_str)
        if is_selected_match:
            is_selected_str = is_selected_match.group(1).strip()
            is_selected = [int(x) for x in is_selected_str.split(',')]
        else:
            return None

        # Extract passage_text (this is more complicated due to nested quotes)
        passage_text_match = re.search(
            passage_text_pattern, passage_str, re.DOTALL)
        if passage_text_match:
            # We need to properly parse the array of strings
            passage_arr_str = passage_text_match.group(1)

            # Fix single quotes to make it valid Python
            fixed_str = passage_arr_str.replace(
                "\\n", "\\\\n").replace("\\'", "\\\\'")

            # Split by clearly identifiable patterns
            passages = []
            parts = re.split(r',\s*(?=\')', fixed_str)

            for part in parts:
                # Clean up the part
                cleaned = part.strip()
                if cleaned.startswith("'") and (cleaned.endswith("'") or "'" in cleaned[1:]):
                    # Extract content between first and last quote
                    first_quote = cleaned.find("'")
                    if first_quote >= 0:
                        # Find the last quote that's not escaped
                        last_quote_pos = -1
                        i = len(cleaned) - 1
                        while i > first_quote:
                            if cleaned[i] == "'" and cleaned[i-1] != "\\":
                                last_quote_pos = i
                                break
                            i -= 1

                        if last_quote_pos > first_quote:
                            content = cleaned[first_quote+1:last_quote_pos]
                            passages.append(content)

            # If we couldn't parse properly, try a simpler approach with eval
            if not passages:
                try:
                    # Try to evaluate as Python list
                    passages = ast.literal_eval(f"[{passage_arr_str}]")
                except:
                    # If that fails, just use regular expressions
                    passages = re.findall(r"'([^']*)'", passage_arr_str)

            # Return the extracted data
            return {
                'is_selected': is_selected,
                'passage_text': passages
            }

        return None
    except Exception as e:
        print(f"Error extracting arrays: {e}")
        return None


def generate_ms_marco_triples(data_file, output_file, num_hard_negatives=3):
    """
    Generate triples specifically for MS Marco dataset format.

    Args:
        data_file: Path to input CSV file
        output_file: Path to output triples CSV
        num_hard_negatives: Number of hard negatives per positive passage
    """
    print(f"Loading data from {data_file}...")
    data = pd.read_csv(data_file)

    print(f"Found {len(data)} rows in dataset")
    print(f"Columns: {data.columns.tolist()}")

    # Storage for extracted passages and triples
    all_passages = {}  # query_id -> {positives: [], negatives: []}
    triples = []

    # First pass: Extract passages
    print("Extracting passages...")
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        query_id = row['query_id']
        query = row['query']

        if 'passages' not in data.columns:
            continue

        passage_str = row['passages']
        if not isinstance(passage_str, str):
            continue

        # Extract arrays using our custom function
        extracted = extract_arrays_from_string(passage_str)

        if extracted and 'is_selected' in extracted and 'passage_text' in extracted:
            is_selected = extracted['is_selected']
            passage_text = extracted['passage_text']

            # Validate and truncate if necessary
            min_len = min(len(is_selected), len(passage_text))
            is_selected = is_selected[:min_len]
            passage_text = passage_text[:min_len]

            # Store passages
            all_passages[query_id] = {
                'query': query,
                'positives': [passage_text[i] for i in range(min_len) if is_selected[i] == 1],
                'negatives': [passage_text[i] for i in range(min_len) if is_selected[i] == 0]
            }

    # Count extracted passages
    total_positives = sum(len(p['positives']) for p in all_passages.values())
    total_negatives = sum(len(p['negatives']) for p in all_passages.values())

    print(f"Extracted passages for {len(all_passages)}/{len(data)} queries")
    print(
        f"Total passages: {total_positives} positives, {total_negatives} negatives")

    # Display some examples
    if all_passages:
        print("\nExample passages:")
        example_count = 0
        for query_id, passages in all_passages.items():
            if passages['positives']:
                print(f"Query ID: {query_id}")
                print(f"Query: {passages['query']}")
                print(f"Positive passage: {passages['positives'][0][:100]}...")
                if passages['negatives']:
                    print(
                        f"Negative passage: {passages['negatives'][0][:100]}...")
                print()

                example_count += 1
                if example_count >= 3:
                    break

    # Second pass: Generate triples
    print("Generating triples...")
    for query_id, passages in tqdm(all_passages.items(), total=len(all_passages)):
        query = passages['query']
        positives = passages['positives']
        negatives = passages['negatives']

        # Skip if no positive passages
        if not positives:
            continue

        # For each positive passage
        for positive in positives:
            # If we have negatives, use them
            if negatives:
                # Calculate similarity to find hard negatives
                vectorizer = TfidfVectorizer(stop_words='english')
                texts = [query] + negatives

                try:
                    tfidf_matrix = vectorizer.fit_transform(texts)
                    similarities = cosine_similarity(
                        tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

                    # Sort by similarity (highest first = hardest negatives)
                    negative_indices = similarities.argsort()[::-1]

                    # Take top N
                    used_indices = []
                    for i in range(min(num_hard_negatives * 2, len(negatives))):
                        idx = negative_indices[i]
                        negative = negatives[idx]
                        similarity = similarities[idx]

                        # Skip if negative is the same as positive
                        if negative == positive:
                            continue

                        if len(used_indices) < num_hard_negatives:
                            triple = {
                                'query_id': query_id,
                                'query': query,
                                'positive_doc': positive,
                                'negative_doc': negative,
                                'similarity_score': float(similarity)
                            }

                            triples.append(triple)
                            used_indices.append(idx)

                        if len(used_indices) >= num_hard_negatives:
                            break
                except:
                    # If TF-IDF fails, just use random sampling
                    sample_count = 0
                    max_attempts = min(num_hard_negatives * 3, len(negatives))

                    # Try up to max_attempts to find valid negatives
                    for _ in range(max_attempts):
                        if sample_count >= num_hard_negatives:
                            break

                        negative = random.choice(negatives)

                        # Skip if negative is the same as positive
                        if negative == positive:
                            continue

                        triple = {
                            'query_id': query_id,
                            'query': query,
                            'positive_doc': positive,
                            'negative_doc': negative,
                            'similarity_score': 0.0
                        }

                        triples.append(triple)
                        sample_count += 1
            else:
                # No negatives for this query, use negatives from other queries
                other_negatives = []

                for other_id, other_passages in all_passages.items():
                    if other_id != query_id and other_passages['negatives']:
                        other_negatives.extend(other_passages['negatives'])

                # Randomly sample from other negatives
                if other_negatives:
                    # Create a set of used negatives to avoid duplicates
                    used_negatives = set()
                    sample_count = 0
                    max_attempts = min(num_hard_negatives *
                                       3, len(other_negatives))

                    # Try to get unique negatives that aren't the same as positive
                    for _ in range(max_attempts):
                        if sample_count >= num_hard_negatives:
                            break

                        negative = random.choice(other_negatives)

                        # Skip if negative is the same as positive or already used
                        if negative == positive or negative in used_negatives:
                            continue

                        used_negatives.add(negative)

                        triple = {
                            'query_id': query_id,
                            'query': query,
                            'positive_doc': positive,
                            'negative_doc': negative,
                            'similarity_score': 0.0
                        }

                        triples.append(triple)
                        sample_count += 1

    # Handle queries with no extracted passages
    print("Handling queries with no extracted passages...")
    queries_with_passages = set(all_passages.keys())
    queries_without_passages = set(
        data['query_id'].tolist()) - queries_with_passages

    if queries_without_passages:
        # Get all queries for negative sampling
        all_queries = data['query'].tolist()

        for idx, row in tqdm(data.iterrows(), total=len(data)):
            query_id = row['query_id']

            if query_id in queries_without_passages:
                query = row['query']

                # Use query as positive (last resort)
                positive = query

                # Sample other queries as negatives
                other_queries = [q for q in all_queries if q != query]

                if other_queries:
                    # Track used negatives to avoid duplicates
                    used_negatives = set()
                    sample_count = 0
                    max_attempts = min(num_hard_negatives *
                                       3, len(other_queries))

                    # Try to get unique negatives
                    for _ in range(max_attempts):
                        if sample_count >= num_hard_negatives:
                            break

                        negative = random.choice(other_queries)

                        # Skip if already used
                        if negative in used_negatives:
                            continue

                        used_negatives.add(negative)

                        triple = {
                            'query_id': query_id,
                            'query': query,
                            'positive_doc': positive,
                            'negative_doc': negative,
                            'similarity_score': 0.0
                        }

                        triples.append(triple)
                        sample_count += 1

    # Save to CSV
    print(f"Generated {len(triples)} initial triples")

    # Validate and filter triples
    triples = validate_and_filter_triples(triples)

    triples_df = pd.DataFrame(triples)
    triples_df.to_csv(output_file, index=False)
    print(f"Saved {len(triples_df)} validated triples to {output_file}")

    return triples_df


def process_all_datasets():
    """Process all MS Marco datasets."""
    # File paths
    input_files = {
        'train': 'msmarco_train_cleaned.csv',
        'valid': 'msmarco_validation_cleaned.csv',
        'test': 'msmarco_test_cleaned.csv'
    }

    # Process each file
    for split, input_file in input_files.items():
        output_file = f'{split}_triples.csv'

        print(f"\nProcessing {split} dataset...")
        generate_ms_marco_triples(input_file, output_file)

    print("\nAll datasets processed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate triples for MS Marco dataset")
    parser.add_argument("--file", type=str, help="Process a specific file")
    parser.add_argument("--output", type=str, help="Output file name")
    parser.add_argument("--negatives", type=int, default=3,
                        help="Number of hard negatives per positive")

    args = parser.parse_args()

    if args.file:
        output = args.output or f"triples_{args.file.split('/')[-1]}"
        generate_ms_marco_triples(args.file, output, args.negatives)
    else:
        process_all_datasets()
