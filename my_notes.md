
# GPT-2 Implementation Notes

---

## Q: What are `last_hidden_state` and `last_token` in `GPT2Model.forward()`?

`GPT2Model.forward()` returns a dictionary:
```python
return {'last_hidden_state': sequence_output, 'last_token': last_token}
```

### `last_hidden_state` — Representations for ALL tokens

**Shape:** `[batch_size, seq_len, 768]`

Consider a batch of 2 sentences:
- Sentence 1: `"I love this movie"` → 4 tokens, no padding
- Sentence 2: `"Good"` → 1 token + 3 padding tokens

```
           token0    token1    token2    token3
            "I"     "love"    "this"   "movie"
sentence1 [ [768d]   [768d]   [768d]   [768d] ]
sentence2 [ [768d]   [768d]   [768d]   [768d] ]
           "Good"   <pad>     <pad>    <pad>
```

Each cell is a 768-dimensional vector representing that token **after being processed by all 12 Transformer layers**. It encodes contextual information from the token itself and all tokens to its left.

---

### `last_token` — Representation of the last VALID token per sentence

**Shape:** `[batch_size, 768]`

```
           token0    token1    token2    token3
            "I"     "love"    "this"   "movie"
sentence1 [ [768d]   [768d]   [768d]  ►[768d]◄ ]  ← picks token3: "movie"
sentence2 [ [768d]  ►[768d]◄  [768d]   [768d]  ]  ← picks token0: "Good" (last non-pad)
            "Good"  <pad>     <pad>    <pad>

Result:
last_token = [ [768d],   ← "movie" vector for sentence 1
               [768d] ]  ← "Good"  vector for sentence 2
```

---

### Why use `last_token` for classification?

GPT-2 is a **left-to-right** autoregressive model — each token can only attend to tokens to its left:

```
"I"     sees: "I"
"love"  sees: "I love"
"this"  sees: "I love this"
"movie" sees: "I love this movie"  ← has seen the full sentence!
```

The last valid token's vector **accumulates information from the entire sentence**, making it the best representation for sentence-level tasks like sentiment classification.

This is analogous to how BERT uses the `[CLS]` token — but in the opposite direction (BERT is bidirectional; GPT-2 is left-to-right).

---

### Corresponding code in `models/gpt2.py`

```python
# Compute index of last non-padding token per sentence
last_non_pad_idx = attention_mask.sum(dim=1) - 1
# attention_mask = [[1,1,1,1], [1,0,0,0]]
# .sum(dim=1)    = [4, 1]
# - 1            = [3, 0]  ← sentence1 takes index 3, sentence2 takes index 0

last_token = sequence_output[torch.arange(batch_size), last_non_pad_idx]
# sentence1: sequence_output[0, 3] → 768-dim vector for "movie"
# sentence2: sequence_output[1, 0] → 768-dim vector for "Good"
```

---

### Summary

| | `last_hidden_state` | `last_token` |
|--|--|--|
| Shape | `[bs, seq_len, 768]` | `[bs, 768]` |
| Content | Vector for **every** token | Vector for the **last valid** token per sentence |
| Use case | Token-level tasks (e.g. NER) | Sentence-level tasks (e.g. sentiment classification) ← use this |