import numpy as np
import re
import json
from collections import Counter

file = "data.txt"
special_tokens = ["<PAD>", "<UNK>"]

lines = []
with open(file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 2000:
            break
        lines.append(line.lower())

text = ''.join(lines)

tokens = re.findall(r"\b\w+\b|[.,!?;:]", text)
freq = Counter(tokens)

# we build the vocabulary with words which appear atleast 5 times in the corpus
vocab_tokens = []
for token, count in freq.items():
    if count>=3:
        vocab_tokens.append(token)
        
idx_to_word = special_tokens + sorted(vocab_tokens)
word_to_idx = {}
for idx, word in enumerate(idx_to_word):
    word_to_idx[word] = idx

idx_to_word_dict = {idx: word for idx, word in enumerate(idx_to_word)}
vocab = {
    "word_to_idx": word_to_idx,
    "idx_to_word": idx_to_word_dict
}
print(len(word_to_idx))

with open("vocab.json", "w") as f:
    json.dump(vocab, f)
    
#now lets encode all the tokens
encoded_text = []
unk_idx = word_to_idx["<UNK>"]
for token in tokens:
    idx = word_to_idx.get(token, unk_idx)
    encoded_text.append(idx)
np.save("encoded_tokens.npy", np.array(encoded_text, dtype=np.int32))
