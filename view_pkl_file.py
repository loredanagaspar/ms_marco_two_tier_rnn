import random
import pickle

with open("tokenized/train_tokenized.pkl", "rb") as f:
    train_data = pickle.load(f)

triplets = list(zip(
    train_data["tokenized_queries"],
    train_data["tokenized_positives"],
    train_data["tokenized_negatives"]
))

print("\n🔍 Inspecting 5 random training triplets:")
for i in random.sample(range(len(triplets)), 5):
    q, pos, neg = triplets[i]
    print(f"\nTriplet {i}:")
    print(f"  Query   : {' '.join(q[:20])}...")
    print(f"  Positive: {' '.join(pos[:30])}...")
    print(f"  Negative: {' '.join(neg[:30])}...")
