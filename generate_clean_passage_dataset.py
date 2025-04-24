from datasets import load_dataset
import pandas as pd
# Rebuild and cache raw passages from Hugging Face if not available
import os
if not os.path.exists("ms_marco_passages_cleaned.csv"):
    print("Loading MS MARCO passage corpus from HuggingFace...")
    dataset = load_dataset("ms_marco", "v1.1", split="train")

    print("Extracting passage texts...")
    all_passages = []
    for row in dataset:
        passages = row.get("passages", {})
        if isinstance(passages, dict) and "passage_text" in passages:
            all_passages.extend(passages["passage_text"])

    df = pd.DataFrame({"passage_text": all_passages})
    df.to_csv("ms_marco_passages_cleaned.csv", index=False)
    print(f"Saved {len(all_passages)} passages to ms_marco_passages_cleaned.csv")