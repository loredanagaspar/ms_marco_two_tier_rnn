# file: generate_triplet_embeddings.py
import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from torch.nn.utils.rnn import pad_sequence
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def token_vectors(tokens, model):
    return [torch.tensor(model.wv[word]) for word in tokens if word in model.wv]


def process_triplet_file(filepath, model, output_prefix):
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} triplets from {filepath}")

    if "query_id" in df.columns:
        os.makedirs("triplet_embeddings", exist_ok=True)
        df["query_id"].to_csv(
            f"triplet_embeddings/{output_prefix}_query_ids.csv", index=False)
        df = df.drop(columns=["query_id"])
    if "similarity_score" in df.columns:
        df = df.drop(columns=["similarity_score"])

    query_tensor_list = []
    pos_tensor_list = []
    neg_tensor_list = []

    for i, row in df.iterrows():
        query = row["query"].lower().split()
        pos = row.get("positive", row.get("positive_doc", "")).lower().split()
        neg = row.get("negative", row.get("negative_doc", "")).lower().split()

        q_vec = token_vectors(query, model)
        p_vec = token_vectors(pos, model)
        n_vec = token_vectors(neg, model)

        if q_vec and p_vec and n_vec:
            query_tensor_list.append(torch.stack(q_vec))
            pos_tensor_list.append(torch.stack(p_vec))
            neg_tensor_list.append(torch.stack(n_vec))
        else:
            print(
                f"Skipping row {i} due to empty tensor (possibly all tokens out of vocab)")

    query_batch = pad_sequence(query_tensor_list, batch_first=True)
    pos_batch = pad_sequence(pos_tensor_list, batch_first=True)
    neg_batch = pad_sequence(neg_tensor_list, batch_first=True)

    os.makedirs("triplet_embeddings", exist_ok=True)
    torch.save(query_batch, f"triplet_embeddings/{output_prefix}_query.pt")
    torch.save(pos_batch, f"triplet_embeddings/{output_prefix}_positive.pt")
    torch.save(neg_batch, f"triplet_embeddings/{output_prefix}_negative.pt")

    print(f"Saved {output_prefix} triplet embeddings to 'triplet_embeddings/'")


def visualize_top_embeddings(model, top_n=100):
    words = model.wv.index_to_key[:top_n]
    vectors = np.array([model.wv[word] for word in words])

    tsne = TSNE(n_components=2, random_state=42,
                init='pca', learning_rate='auto')
    reduced = tsne.fit_transform(vectors)

    plt.figure(figsize=(12, 10))
    for i, word in enumerate(words):
        x, y = reduced[i]
        plt.scatter(x, y)
        plt.annotate(word, (x, y), fontsize=9)
    plt.title(f"t-SNE of Top {top_n} Word Embeddings")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    model = Word2Vec.load("models/word2vec_combined.bin")

    process_triplet_file("train_triples.csv", model, "train")
    process_triplet_file("valid_triples.csv", model, "valid")
    process_triplet_file("test_triples.csv", model, "test")

    visualize_top_embeddings(model, top_n=100)


if __name__ == "__main__":
    main()
