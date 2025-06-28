import json
import re
from collections import Counter

def tokenize(formula):
    return [token for token in formula.strip().split() if token]

def load_formulas(file_path):
    formulas = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            f = line.strip()
            if f:
                formulas[i] = f
    print(f"Loaded {len(formulas)} formulas from {file_path}")
    return formulas

def create_vocab(formulas, min_freq=1):
    #print("Building vocabulary...")
    counter = Counter()
    for i, f in formulas.items():
        try:
            counter.update(tokenize(f))
        except Exception as e:
            print(f"Error on formula {i}: {f}")
            raise e
    vocab = [t for t, c in counter.items() if c >= min_freq]
    print(f"Vocabulary size: {len(vocab)} (from {len(counter)} unique tokens)")
    return sorted(vocab), counter

def map_tokens(vocab, special=['<PAD>', '<START>', '<END>', '<UNK>']):
    token_to_id = {t: i for i, t in enumerate(special)}
    for t in vocab:
        if t not in token_to_id:
            token_to_id[t] = len(token_to_id)
    id_to_token = {i: t for t, i in token_to_id.items()}
    return token_to_id, id_to_token

def encode_formula(f, token_to_id, max_len=150):
    tokens = tokenize(f)
    ids = [token_to_id['<START>']] + [token_to_id.get(t, token_to_id['<UNK>']) for t in tokens] + [token_to_id['<END>']]
    if len(ids) > max_len:
        return ids[:max_len]
    return ids + [token_to_id['<PAD>']] * (max_len - len(ids))

def decode_ids(ids, id_to_token):
    pad_id = next((i for i, t in id_to_token.items() if t == '<PAD>'), None)
    end_id = next((i for i, t in id_to_token.items() if t == '<END>'), None)
    start_id = next((i for i, t in id_to_token.items() if t == '<START>'), None)
    tokens = []
    for i in ids:
        if i in (pad_id, end_id):
            break
        if i == start_id:
            continue
        tokens.append(id_to_token.get(i, '<UNK>'))
    return ' '.join(tokens)

def save_vocab(token_to_id, id_to_token, counts, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({
            'token_to_id': token_to_id,
            'id_to_token': {str(i): t for i, t in id_to_token.items()},
            'token_counts': dict(counts),
            'vocab_size': len(token_to_id)
        }, f, indent=2, ensure_ascii=False)
    print(f"Vocabulary saved to {path}")

def load_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    t2i = data['token_to_id']
    i2t = {int(i): t for i, t in data['id_to_token'].items()}
    counts = Counter(data.get('token_counts', {}))
    print(f"Vocabulary loaded from {path}")
    print(f"Size: {len(t2i)} tokens")
    return t2i, i2t, counts

def clean_vocab(original):
    specials = ["<PAD>", "<START>", "<END>", "<UNK>"]
    patterns = {
        'prefix': {
            "\\text", "\\bf", "\\it", "\\Large", "\\LARGE", "\\huge", "\\small",
            "\\emph", "\\rm", "\\sf", "\\tt", "\\sc", "\\boldmath", "\\mathversion",
            "\\scriptsize", "\\displaystyle", "\\ensuremath", "\\normalsize",
            "\\protect", "\\everymath"
        },
        'substring': {
            "\\protect", "\\def", "\\newcommand", "\\let", "\\expandafter",
            "\\relax", "\\noalign", "\\nonumber", "\\null", "\\mathversion",
            "\\leavevmode", "\\label", "\\hbox", "\\framebox"
        }
    }

    def remove(t, unesc):
        if any(t.startswith(p) for p in patterns['prefix']):
            return True
        if any(p in t for p in patterns['substring']):
            return True
        if re.match(r"^{.*}$", t) or re.match(r"\{.+\}", t):
            return True
        if t.startswith("\\\\") and t[1:] in unesc:
            return True
        return False

    unesc = {t for t in original if not t.startswith("\\\\")}
    cleaned = [t for t in original if t not in specials and not remove(t, unesc)]
    final = specials + cleaned
    token_to_id = {t: i for i, t in enumerate(final)}
    id_to_token = {str(i): t for t, i in token_to_id.items()}
    return {
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "vocabulary_size": len(token_to_id)
    }

def build_and_save_vocab(formulas_file, output_file, min_freq=1):
    formulas = load_formulas(formulas_file)
    vocab, counts = create_vocab(formulas, min_freq)
    token_to_id, id_to_token = map_tokens(vocab)
    save_vocab(token_to_id, id_to_token, counts, output_file)
    return token_to_id, id_to_token, counts

def clean_saved_vocab(input_file, output_file):
    #print(f"Loading from {input_file}...")
    with open(input_file, "r", encoding='utf-8') as f:
        original = json.load(f)["token_to_id"]
    #print(f"Original size: {len(original)}")
    cleaned = clean_vocab(original)
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    removed = len(original) - cleaned['vocabulary_size']
    print(f"Saved cleaned vocab to {output_file}")
    print(f"Size after cleaning: {cleaned['vocabulary_size']}, removed: {removed}")
    #print(f"<PAD> index: {cleaned['token_to_id']['<PAD>']}")
    return cleaned

#change the directories while using
if __name__ == "__main__":
    formulas_path = "path to formulas"
    raw_vocab_path = "building initial vocabulary"
    vocab_input_path = "same as above"
    cleaned_vocab_path = "path for the cleaned vocab"

    print("=== Building Vocabulary ===")
    token_to_id, id_to_token, counts = build_and_save_vocab(
        formulas_path, raw_vocab_path, min_freq=1
    )

    print("\n=== Cleaning Vocabulary ===")
    cleaned = clean_saved_vocab(vocab_input_path, cleaned_vocab_path)
