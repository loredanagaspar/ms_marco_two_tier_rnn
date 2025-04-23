from datasets import load_dataset


# Load MS Marco v1.1 from Hugging Face
dataset = load_dataset("ms_marco", "v1.1")
print(dataset)
print(dataset.keys())

# Process and save each split with only needed columns
for split in ['train', 'validation', 'test']:
    if split in dataset:
        print(f"\n=== {split.upper()} SPLIT ===")
        df = dataset[split].to_pandas()

        # Drop only irrelevant fields
        df.drop(columns=['answers', 'query_type',
                'wellFormedAnswers'], inplace=True, errors='ignore')

        # Ensure passages are serialized as strings for CSV compatibility
        df['passages'] = df['passages'].apply(
            lambda x: str(x) if isinstance(x, (list, dict)) else x)

        # Save full cleaned data for future use (positives + negatives)
        csv_path = f"msmarco_{split}_cleaned.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved full {split} data to {csv_path}")

        # 1print number of rows
        print(f"Saved full {split} data to {csv_path} with {len(df)} rows")
        # Show schema and sample
        print(f"Schema:\n{df.dtypes}")
        print("\nSample rows:")
        print(df[['query', 'passages']].head(1))
