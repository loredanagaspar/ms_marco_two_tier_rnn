# evaluate_best_model.py
import torch
import os
from two_tower_rnn import RNNEncoder, build_embedding_matrix, simple_tokenize, process_text, triplet_loss


def evaluate_model(test_triples, encoder_q, encoder_d, word2idx):
    encoder_q.eval()
    encoder_d.eval()
    total_loss = 0
    with torch.no_grad():
        for triple in test_triples:
            padded_ids = process_text(triple, word2idx)
            q = encoder_q(padded_ids[0].unsqueeze(0))
            pos = encoder_d(padded_ids[1].unsqueeze(0))
            neg = encoder_d(padded_ids[2].unsqueeze(0))
            loss = triplet_loss(q, pos, neg)
            total_loss += loss.item()
    return total_loss / len(test_triples)


if __name__ == "__main__":
    # Define your test triples
    test_triples = [
        [
            "Where is Rome located?",
            "Rome is the capital of Italy.",
            "The Sahara desert is the largest hot desert in the world."
        ]
    ]

    # Build vocab and embedding matrix
    all_text = " ".join([t for triple in test_triples for t in triple])
    vocab = list(set(simple_tokenize(all_text)))
    embedding_matrix, word2idx = build_embedding_matrix(vocab)

    # Load models
    encoder_q = RNNEncoder(embedding_matrix)
    encoder_d = RNNEncoder(embedding_matrix)

    encoder_q.load_state_dict(torch.load("models/best_encoder_q.pt"))
    encoder_d.load_state_dict(torch.load("models/best_encoder_d.pt"))

    test_loss = evaluate_model(test_triples, encoder_q, encoder_d, word2idx)
    print(f"Test Loss: {test_loss:.4f}")
