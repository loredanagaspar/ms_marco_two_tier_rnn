# two_tower_rnn.py

import gensim.downloader as api
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import re
import random
import os
import wandb
import pandas as pd
import pickle
from dotenv import load_dotenv
load_dotenv()

# Set device for GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Load pretrained Word2Vec embeddings
glove_vectors = api.load("word2vec-google-news-300")
embedding_dim = glove_vectors.vector_size

# Step 2: Simple whitespace + punctuation tokenizer


def simple_tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

# Step 3: Build vocabulary and embedding matrix from saved vocab file


def load_vocab_and_build_matrix(vocab_path):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    print(f"Loaded vocabulary of size {len(vocab)}")
    embedding_matrix = []
    word2idx = {}
    for i, word in enumerate(vocab):
        if word in glove_vectors:
            embedding_matrix.append(torch.tensor(glove_vectors[word]))
        else:
            embedding_matrix.append(torch.randn(embedding_dim))
        word2idx[word] = i
    return torch.stack(embedding_matrix), word2idx

# Step 4: RNN Encoder


class RNNEncoder(nn.Module):
    def __init__(self, embedding_matrix):
        super().__init__()
        _, embedding_dim = embedding_matrix.size()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=True)
        self.rnn = nn.GRU(embedding_dim, 128, batch_first=True)

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)
        _, hidden = self.rnn(embedded)
        return hidden.squeeze(0)

# Step 5: Processing a single (query, doc_pos, doc_neg) triple (already tokenized)


def process_tokenized_tokens(triple_tokens, word2idx):
    token_ids = []
    for tokens in triple_tokens:
        indices = [word2idx.get(token, 0) for token in tokens]
        token_ids.append(torch.tensor(indices))
    return pad_sequence(token_ids, batch_first=True)

# Step 6: Triplet Loss


def triplet_loss(query, pos, neg, margin=1.0):
    pos_dist = F.pairwise_distance(query, pos, p=2)
    neg_dist = F.pairwise_distance(query, neg, p=2)
    return torch.mean(F.relu(pos_dist - neg_dist + margin))


# Step 7: Load real MS MARCO triples using pandas
def load_triples_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df[["query", "positive_doc", "negative_doc"]].values.tolist()

# Step 8: Load pre-tokenized MS MARCO triples from pickle file


def load_tokenized_triples_from_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    queries = data["tokenized_queries"]
    positives = data["tokenized_positives"]
    negatives = data["tokenized_negatives"]
    return list(zip(queries, positives, negatives))

# Step 9: Training loop with validation, early stopping, best model saving, and wandb logging


def evaluate(val_triples, encoder_q, encoder_d, word2idx):
    encoder_q.eval()
    encoder_d.eval()
    total_loss = 0
    with torch.no_grad():
        for triple in val_triples:
            padded_ids = process_tokenized_tokens(triple, word2idx)
            q = encoder_q(padded_ids[0].unsqueeze(0).to(device))
            pos = encoder_d(padded_ids[1].unsqueeze(0).to(device))
            neg = encoder_d(padded_ids[2].unsqueeze(0).to(device))
            loss = triplet_loss(q, pos, neg)
            total_loss += loss.item()
    return total_loss / len(val_triples)


def evaluate_test(test_triples, encoder_q, encoder_d, word2idx):
    encoder_q.eval()
    encoder_d.eval()
    total_loss = 0
    cos_pos_all, cos_neg_all = [], []
    with torch.no_grad():
        for triple in test_triples:
            padded_ids = process_tokenized_tokens(triple, word2idx)
            q = encoder_q(padded_ids[0].unsqueeze(0).to(device))
            pos = encoder_d(padded_ids[1].unsqueeze(0).to(device))
            neg = encoder_d(padded_ids[2].unsqueeze(0).to(device))
            loss = triplet_loss(q, pos, neg)
            total_loss += loss.item()
            cos_pos = F.cosine_similarity(q, pos)
            cos_neg = F.cosine_similarity(q, neg)
            cos_pos_all.append(cos_pos.item())
            cos_neg_all.append(cos_neg.item())

    test_loss = total_loss / len(test_triples)
    wandb.log({
        "test_loss": test_loss,
        "cos_sim/positive": wandb.Histogram(cos_pos_all),
        "cos_sim/negative": wandb.Histogram(cos_neg_all)
    })
    print(f"Test Loss: {test_loss:.4f}")


def train(triples, encoder_q, encoder_d, word2idx, epochs=5, lr=1e-3, val_triples=None, test_triples=None, patience=2, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    encoder_q.to(device)
    encoder_d.to(device)
    optimizer = torch.optim.Adam(
        list(encoder_q.parameters()) + list(encoder_d.parameters()), lr=lr)
    encoder_q.train()
    encoder_d.train()

    wandb.init(project="two-tower-rnn", config={"epochs": epochs, "lr": lr})
    wandb.watch([encoder_q, encoder_d], log='all')

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(triples)
        for triple in triples:
            padded_ids = process_tokenized_tokens(triple, word2idx)
            q = encoder_q(padded_ids[0].unsqueeze(0).to(device))
            pos = encoder_d(padded_ids[1].unsqueeze(0).to(device))
            neg = encoder_d(padded_ids[2].unsqueeze(0).to(device))
            loss = triplet_loss(q, pos, neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(triples)
        val_loss = evaluate(val_triples, encoder_q, encoder_d,
                            word2idx) if val_triples else None

        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}" +
              (f", Val Loss: {val_loss:.4f}" if val_loss is not None else ""))

        wandb.log({"loss/train": avg_train_loss,
                  "loss/val": val_loss, "epoch": epoch + 1})

        if val_triples:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(encoder_q.state_dict(), os.path.join(
                    save_dir, "best_encoder_q.pt"))
                torch.save(encoder_d.state_dict(), os.path.join(
                    save_dir, "best_encoder_d.pt"))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    if test_triples:
        evaluate_test(test_triples, encoder_q, encoder_d, word2idx)


# MAIN FUNCTION
if __name__ == "__main__":

    # Paths to your prepared data
    train_path = "tokenized/train_tokenized.pkl"
    val_path = "tokenized/valid_tokenized.pkl"
    test_path = "tokenized/test_tokenized.pkl"
    vocab_path = "vocab.pkl"

    # Load vocab and embedding matrix
    embedding_matrix, word2idx = load_vocab_and_build_matrix(vocab_path)

    # Load tokenized triples
    train_triples = load_tokenized_triples_from_pickle(train_path)
    val_triples = load_tokenized_triples_from_pickle(val_path)
    test_triples = load_tokenized_triples_from_pickle(test_path)

    # Init encoders
    query_encoder = RNNEncoder(embedding_matrix)
    doc_encoder = RNNEncoder(embedding_matrix)

    # Start training
    train(train_triples, query_encoder, doc_encoder, word2idx,
          val_triples=val_triples, test_triples=test_triples)
