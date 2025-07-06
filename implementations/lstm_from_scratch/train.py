import numpy as np
import json

context_size = 50
chunk_size = 10
encoded_data = np.load("encoded_tokens.npy")

with open("vocab.json", "r") as f:
    vocab = json.load(f)

word_to_idx = vocab["word_to_idx"]
idx_to_word = vocab["idx_to_word"]
idx_to_word = {int(k): v for k, v in idx_to_word.items()}

split_idx = int(0.9*len(encoded_data))
train_data = encoded_data[:split_idx]
test_data = encoded_data[split_idx:]

def sequence_chunks(data, seq_len, stride):
    inputs, targets = [], []
    for i in range(0, len(data) - seq_len, stride):
        inputs.append(data[i:i+seq_len])
        targets.append(data[i+1:i+seq_len+1])
    return inputs, targets

train_inputs, train_targets = sequence_chunks(train_data, context_size, stride=5)

embed_dim = 32
hidden_dim = 64
vocab_size = len(word_to_idx)
E = np.random.randn(vocab_size, embed_dim) * 0.01

def init_lstm_weights(embed_dim=32, hidden_dim=64, vocab_size=5000):
    def xavier(input_dim, output_dim):
        lim = np.sqrt(6/(input_dim+output_dim))
        return np.random.uniform(-lim, lim, size=(input_dim, output_dim))

    concat_size = embed_dim + hidden_dim
    return {
    "Wf": np.random.randn(hidden_dim, concat_size) * 0.01, "bf": np.zeros((hidden_dim, 1)),
    "Wi": np.random.randn(hidden_dim, concat_size) * 0.01, "bi": np.zeros((hidden_dim, 1)),
    "Wo": np.random.randn(hidden_dim, concat_size) * 0.01, "bo": np.zeros((hidden_dim, 1)),
    "Wc": np.random.randn(hidden_dim, concat_size) * 0.01, "bc": np.zeros((hidden_dim, 1)),
    "Wout": np.random.randn(vocab_size, hidden_dim) * 0.01, "bout": np.zeros((vocab_size, 1)),
    }

W = init_lstm_weights(embed_dim, hidden_dim, vocab_size)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1/(1+np.exp(-x))

def softmax(x):
    x = np.clip(x, -500, 500)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0, keepdims=True)

def lstm_forward(x_seq, h0, c0, W, E):
    h = h0.copy()
    c = c0.copy()
    cache = []

    for idx in x_seq:
        x = E[idx].reshape(-1, 1)
        xh = np.concatenate([x, h], axis=0)

        f = sigmoid(W['Wf'] @ xh + W['bf'])
        i = sigmoid(W['Wi'] @ xh + W['bi'])
        o = sigmoid(W['Wo'] @ xh + W['bo'])
        c_tilde = np.tanh(W['Wc'] @ xh + W['bc'])

        c_prev = c.copy()
        c = f * c + i * c_tilde
        h = o * np.tanh(c)

        cache.append({
            'xh': xh, 'h': h, 'c': c, 'c_prev': c_prev,
            'f': f, 'i': i, 'o': o, 'c_tilde': c_tilde, 'x_idx': idx
        })
    return cache

def output(h, target_idx, W):
    logits = W['Wout'] @ h + W['bout']
    probs = softmax(logits)
    loss = -np.log(probs[target_idx, 0] + 1e-9)
    return loss, probs, logits

def lstm_backward(cache, y_seq, W, E, probs_list):
    dW = {k: np.zeros_like(v) for k, v in W.items()}
    dE = np.zeros_like(E)
    dh_next = np.zeros((hidden_dim, 1))
    dc_next = np.zeros((hidden_dim, 1))

    for t in reversed(range(len(cache))):
        h = cache[t]['h']
        c = cache[t]['c']
        c_prev = cache[t]['c_prev']
        xh = cache[t]['xh']
        f = cache[t]['f']
        i = cache[t]['i']
        o = cache[t]['o']
        c_tilde = cache[t]['c_tilde']
        x_idx = cache[t]['x_idx']

        probs = probs_list[t]
        dy = probs.copy()
        dy[y_seq[t]] -= 1

        dW['Wout'] += dy @ h.T
        dW['bout'] += dy

        dh = W['Wout'].T @ dy + dh_next

        do = dh * np.tanh(c)
        do_raw = do * o * (1 - o)

        dc = dh * o * (1 - np.tanh(c)**2) + dc_next

        df = dc * c_prev
        df_raw = df * f * (1 - f)

        di = dc * c_tilde
        di_raw = di * i * (1 - i)

        dc_tilde = dc * i
        dc_tilde_raw = dc_tilde * (1 - c_tilde**2)

        d_concat = (
            W['Wf'].T @ df_raw +
            W['Wi'].T @ di_raw +
            W['Wo'].T @ do_raw +
            W['Wc'].T @ dc_tilde_raw
        )

        dx = d_concat[:embed_dim, :]
        dh_next = d_concat[embed_dim:, :]
        dc_next = dc * f

        dE[x_idx] += dx.ravel()
        dW['Wf'] += df_raw @ xh.T
        dW['bf'] += df_raw
        dW['Wi'] += di_raw @ xh.T
        dW['bi'] += di_raw
        dW['Wo'] += do_raw @ xh.T
        dW['bo'] += do_raw
        dW['Wc'] += dc_tilde_raw @ xh.T
        dW['bc'] += dc_tilde_raw

    for k in dW:
        np.clip(dW[k], -5, 5, out=dW[k])
    np.clip(dE, -5, 5, out=dE)

    return dW, dE

learning_rate = 0.1
num_epochs = 10

for epoch in range(num_epochs):
    total_loss = 0
    combined = list(zip(train_inputs, train_targets))
    np.random.shuffle(combined)
    shuffled_inputs, shuffled_targets = zip(*combined)

    for i, (x_seq, y_seq) in enumerate(zip(shuffled_inputs, shuffled_targets)):
        h = np.zeros((hidden_dim, 1))
        c = np.zeros((hidden_dim, 1))

        for t in range(0, len(x_seq), chunk_size):
            x_chunk = x_seq[t:t+chunk_size]
            y_chunk = y_seq[t:t+chunk_size]

            if len(x_chunk) != chunk_size:
                continue

            cache = lstm_forward(x_chunk, h, c, W, E)
            loss = 0
            probs_list = []

            for step in range(chunk_size):
                l, p, _ = output(cache[step]['h'], y_chunk[step], W)
                loss += l
                probs_list.append(p)

            loss /= chunk_size
            total_loss += loss

            dW, dE = lstm_backward(cache, y_chunk, W, E, probs_list)

            for k in W:
                W[k] -= learning_rate * dW[k]
            E -= learning_rate * dE

            h = cache[-1]['h'].copy()
            c = cache[-1]['c'].copy()

        if (i + 1) % 100 == 0:
            avg_loss_so_far = total_loss / ((i + 1) * (context_size // chunk_size))
            print(f"Epoch {epoch+1}, Step {i+1}/{len(train_inputs)}, Avg Loss: {avg_loss_so_far:.4f}")

    avg_epoch_loss = total_loss / (len(train_inputs) * (context_size // chunk_size))
    print(f"Epoch {epoch+1}/{num_epochs}, Avg Epoch Loss: {avg_epoch_loss:.4f}")


def generate_text(W, E, word_to_idx, idx_to_word, seed_text, num_to_generate):
    hidden_dim = W['bf'].shape[0]
    unk_token_idx = -1
    for idx, word in idx_to_word.items():
        if word.lower() == '<unk>':
            unk_token_idx = idx
            break

    h = np.zeros((hidden_dim, 1))
    c = np.zeros((hidden_dim, 1))

    tokenized_seed = [word_to_idx.get(word.lower(), unk_token_idx) for word in seed_text.split()]
    if not tokenized_seed or all(t == unk_token_idx for t in tokenized_seed):
         tokenized_seed = [np.random.randint(0, len(word_to_idx) -1)]

    for token_index in tokenized_seed:
        x = E[token_index].reshape(-1, 1)
        xh = np.concatenate([x, h], axis=0)

        f_gate = sigmoid(W['Wf'] @ xh + W['bf'])
        i_gate = sigmoid(W['Wi'] @ xh + W['bi'])
        o_gate = sigmoid(W['Wo'] @ xh + W['bo'])
        c_tilde = np.tanh(W['Wc'] @ xh + W['bc'])

        c = f_gate * c + i_gate * c_tilde
        h = o_gate * np.tanh(c)

    current_token_index = tokenized_seed[-1]
    generated_word_list = []

    for _ in range(num_to_generate):
        x = E[current_token_index].reshape(-1, 1)
        xh = np.concatenate([x, h], axis=0)

        f_gate = sigmoid(W['Wf'] @ xh + W['bf'])
        i_gate = sigmoid(W['Wi'] @ xh + W['bi'])
        o_gate = sigmoid(W['Wo'] @ xh + W['bo'])
        c_tilde = np.tanh(W['Wc'] @ xh + W['bc'])

        c = f_gate * c + i_gate * c_tilde
        h = o_gate * np.tanh(c)

        logits = W['Wout'] @ h + W['bout']
        probs = softmax(logits)

        next_predicted_token_index = np.random.choice(len(probs), p=probs.ravel())

        if unk_token_idx != -1 and next_predicted_token_index == unk_token_idx:
            sorted_indices_by_prob = np.argsort(probs.ravel())
            best_choice_idx = sorted_indices_by_prob[-1]

            if best_choice_idx != unk_token_idx:
                next_predicted_token_index = best_choice_idx
            else:
                if len(sorted_indices_by_prob) >= 2:
                    next_predicted_token_index = sorted_indices_by_prob[-2]
                else:
                    next_predicted_token_index = np.random.choice([idx for idx in range(len(probs)) if idx != unk_token_idx])

        generated_word_list.append(idx_to_word.get(next_predicted_token_index, "<?>"))
        current_token_index = next_predicted_token_index

    return " ".join(generated_word_list)

SEED_TEXT = "We should"
WORDS_TO_GENERATE = 500

generated_sequence = generate_text(
    W, E, word_to_idx, idx_to_word, SEED_TEXT, WORDS_TO_GENERATE
)

final_output_text = SEED_TEXT + " " + generated_sequence
words_for_display = final_output_text.split()
current_line_length = 0
for word_item in words_for_display:
    if current_line_length + len(word_item) + 1 > 80:
        print()
        current_line_length = 0
    print(word_item, end=" ")
    current_line_length += len(word_item) + 1
print("\n")
