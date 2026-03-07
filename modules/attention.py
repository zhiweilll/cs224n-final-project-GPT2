import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads # h: number of attention heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads) # dk = dv = d/h: attention head size
    self.all_head_size = self.num_attention_heads * self.attention_head_size # d: all head size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    """
    from single head to multi-head
    input x:    [bs, seq_len, 768]
     ↓ linear_layer (W^Q / W^K / W^V)
    proj:      [bs, seq_len, 768]       = [bs, seq_len, h × dk]
        ↓ rearrange 'b t (h d) -> b t h d' (split into heads)
    proj:      [bs, seq_len, 12, 64]    (split into 12 heads)
        ↓ rearrange 'b t h d -> b h t d' (move head dimension to front)
    proj:      [bs, 12, seq_len, 64]    (move head dimension to front)
    """
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):

    """
    key, query, value 均为 [bs, h, seq_len, dk]

    Step 1: Q @ Kᵀ / √dk  →  scores [bs, h, seq_len, seq_len]
    Step 2: + causal mask（torch.triu，未来位置 → -∞）
    Step 3: + attention_mask（padding 位置 → -∞，广播进来）
    Step 4: softmax(scores, dim=-1)  →  attn_weights [bs, h, seq_len, seq_len]
    Step 5: dropout(attn_weights)
    Step 6: attn_weights @ V  →  [bs, h, seq_len, dk]
    Step 7: rearrange 'b h t d -> b t (h d)'  →  [bs, seq_len, 768]
    """
    # key, query, value: [bs, h, seq_len, dk]

    # 1: QKᵀ / √dk ->  [bs, h, seq_len, seq_len]
    dk = self.attention_head_size
    attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (dk ** 0.5) # transpose seq and dk dimension

    # 2. CausalMask before softmax
    seq_len = attention_scores.size(-1)
    causal_mask = torch.triu(
      torch.ones(seq_len, seq_len, device= attention_scores.device, dtype=torch.bool), diagonal=1
    )
    attention_scores = attention_scores.masked_fill(mask=causal_mask, value=float('-inf'))
    attention_scores = attention_scores + attention_mask # [bs, h, seq_len, seq_len]

    # 3. Softmax
    # softmax over column (last seq_len) dimension, scores[i,j] means the similarity between query i and key j
    # In each row, the value is between 0 and 1, and the sum of the row is 1.
    attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1) 

    # 4. Dropout
    attention_weights = self.dropout(attention_weights)

    # 5. attention weights * V ->  [bs, h, seq_len, dk]
    attention_output = torch.matmul(attention_weights, value)

    # 6. Concatenate multi-heads back to original shape -> [bs, seq_len, d]
    attention_output = rearrange(attention_output, 'b h t d -> b t (h d)')

    return attention_output

  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, d]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, d]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
