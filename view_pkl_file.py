import pickle

file_path = "tokenized/test_tokenized.pkl"
with open(file_path, "rb") as f:
    data = pickle.load(f)

# Print the type and keys (if it's a dictionary)
print("Type of data:", type(data))
if isinstance(data, dict):
    print("Keys in data:", data.keys())

# If it's a list of dictionaries, check the first element
if isinstance(data, list) and len(data) > 0:
    print("First element keys:", data[0].keys())
