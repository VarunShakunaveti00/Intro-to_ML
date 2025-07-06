import numpy as np

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
