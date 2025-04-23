import pandas as pd
import re
import sys


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

            # Return the extracted data
            return {
                'is_selected': is_selected,
                'passage_text': passages
            }

        return None
    except Exception as e:
        print(f"Error extracting arrays: {e}")
        return None


def examine_first_query(file_path):
    """
    Examine the first query in the dataset, showing is_selected values
    and corresponding passages.
    """
    try:
        # Load the data
        data = pd.read_csv(file_path)

        if len(data) == 0:
            print("Error: Dataset is empty")
            return

        # Get the first row
        first_row = data.iloc[1]

        print(f"Query ID: {first_row['query_id']}")
        print(f"Query: {first_row['query']}")

        # Extract passages
        if 'passages' not in data.columns:
            print("Error: No 'passages' column found in the dataset")
            return

        passage_str = first_row['passages']
        if not isinstance(passage_str, str):
            print("Error: 'passages' column is not a string")
            return

        # Extract arrays
        extracted = extract_arrays_from_string(passage_str)

        if not extracted:
            print("Error: Failed to extract arrays from passages string")
            return

        is_selected = extracted['is_selected']
        passage_text = extracted['passage_text']

        # Ensure arrays have same length
        min_len = min(len(is_selected), len(passage_text))
        is_selected = is_selected[:min_len]
        passage_text = passage_text[:min_len]

        # Display all passages with their is_selected values
        print(f"\nFound {min_len} passages:")
        print(f"is_selected array: {is_selected}")

        # Count positive and negative passages
        positive_count = sum(1 for x in is_selected if x == 1)
        negative_count = sum(1 for x in is_selected if x == 0)

        print(f"Positive passages (is_selected=1): {positive_count}")
        print(f"Negative passages (is_selected=0): {negative_count}")

        # Print each passage with its is_selected value
        print("\nDetailed passages:")
        for i in range(min_len):
            status = "POSITIVE" if is_selected[i] == 1 else "negative"
            print(f"\nPassage {i+1} [{status}]:")
            print(f"{passage_text[i][:300]}...")

        # Now simulate the negative sampling process for this query
        if positive_count > 0 and negative_count > 0:
            print("\n--- Simulating Negative Sampling ---")

            # Get positive and negative passages
            positives = [passage_text[i]
                         for i in range(min_len) if is_selected[i] == 1]
            negatives = [passage_text[i]
                         for i in range(min_len) if is_selected[i] == 0]

            print(f"\nWould generate triples using:")
            print(f"- Query: {first_row['query']}")

            for i, pos in enumerate(positives):
                print(f"\n- Positive {i+1}: {pos[:100]}...")

                # Show which negatives would be selected (just first 2 for brevity)
                print(f"  Would pair with these negatives:")
                for j, neg in enumerate(negatives[:2]):
                    print(f"  - Negative {j+1}: {neg[:100]}...")

                if len(negatives) > 2:
                    print(f"  - ... and {len(negatives)-2} more negatives")

    except Exception as e:
        print(f"Error examining first query: {e}")


if __name__ == "__main__":
    # Either use file path from command line or default
    file_path = sys.argv[1] if len(
        sys.argv) > 1 else 'msmarco_test_cleaned.csv'
    examine_first_query(file_path)
