"""
VeriLLM: Publicly Verifiable Decentralized LLM Inference from Scratch in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_char_vocab
def build_char_vocab(corpus):
    # TODO: build a char-level vocab dict with 'stoi' and 'itos' fields, ids in sorted order from 0
    if isinstance(corpus, list):
        corpus = ' '.join(corpus)
    corpus = sorted(list(set(corpus)))
    stoi = {c: i for i, c in enumerate(corpus)}
    itos = {i: c for i, c in enumerate(corpus)}
    return {
        'stoi': stoi,
        'itos': itos
    }

# Step 2 - encode_string
def encode_string(text, vocab):
    # TODO: convert text into a list of integer token ids using vocab['stoi'].
    return [vocab['stoi'][c] for c in text]

# Step 3 - decode_ids
def decode_ids(ids, vocab):
    # TODO: decode a sequence of token ids back into the original string using vocab['itos'].
    return ''.join(vocab['itos'][i] for i in ids)

# Step 4 - embed_tokens
import torch

def embed_tokens(token_ids, token_embedding):
    """Look up token embedding vectors for a sequence of token ids.

    Args:
        token_ids: LongTensor of shape (T,).
        token_embedding: FloatTensor of shape (vocab_size, d_model).

    Returns:
        FloatTensor of shape (T, d_model).
    """
    # TODO: select the embedding row for each token id and return (T, d_model)
    return token_embedding[token_ids]

# Step 5 - add_positional_embeddings
import torch

def add_positional_embeddings(token_embeds, pos_embedding, start_pos=0):
    """Add the positional embedding slice [start_pos : start_pos + T] to token_embeds."""
    # TODO: add the appropriate slice of pos_embedding to token_embeds and return the sum.
    T, _ = token_embeds.size()
    return token_embeds + pos_embedding[start_pos:start_pos+T]

# Step 6 - linear_projection
import numpy as np

def linear_projection(x, weight, bias=None):
    """Affine map y = x @ weight + bias used throughout the transformer."""
    # TODO: compute x @ weight and add bias if provided
    bias = 0 if bias is None else bias
    return x @ weight + bias

# Step 7 - compute_attention_scores
def compute_attention_scores(queries, keys):
    # TODO: return the (Tq, Tk) matrix of raw dot-product scores between queries and keys.
    return queries @ keys.T

# Step 8 - scale_attention_scores
import math

def scale_attention_scores(scores, d_head):
    # TODO: scale raw attention scores by 1/sqrt(d_head) for numerical stability.
    return scores / math.sqrt(d_head)

# Step 9 - apply_causal_mask
def apply_causal_mask(scores, query_offset=0):
    # TODO: mask entries where key index > query_offset + query row index with -inf.
    T_q, T_k = scores.shape
    q_indices = np.arange(0, T_q)[:, None] #[T_q, 1]
    k_indices = np.arange(0, T_k)[None, :] #[1, T_k]
    allowed = q_indices + query_offset
    mask = k_indices > allowed
    return np.where(mask, float('-inf'), scores)

# Step 10 - softmax_attention_weights
import numpy as np

def softmax_attention_weights(masked_scores):
    """Convert masked attention scores to a probability distribution via softmax over the last axis."""
    # TODO: apply a numerically stable softmax along the last axis of masked_scores
    max_scores = np.max(masked_scores, axis=-1, keepdims=True)
    adj_masked_score = masked_scores - max_scores
    exp_adj_masked_score = np.exp(adj_masked_score)
    sum_exp_adj_masked_score = np.sum(exp_adj_masked_score, axis=-1, keepdims=True)
    return exp_adj_masked_score / sum_exp_adj_masked_score

# Step 11 - weighted_value_sum
import numpy as np

def weighted_value_sum(attn_weights, values):
    # TODO: combine attention weights (Tq, Tk) with values (Tk, d_head) into context (Tq, d_head).
    return attn_weights @ values

# Step 12 - project_qkv
import numpy as np

def project_qkv(x, attn_params):
    # TODO: project x into query, key, value tensors using attn_params
    bq = attn_params['bq'] if 'bq' in attn_params else None
    bk = attn_params['bk'] if 'bk' in attn_params else None
    bv = attn_params['bv'] if 'bv' in attn_params else None
    q = linear_projection(x, attn_params['Wq'], bq)
    k = linear_projection(x, attn_params['Wk'], bk)
    v = linear_projection(x, attn_params['Wv'], bv)
    return (q, k, v)

# Step 13 - append_kv_cache
def append_kv_cache(kv_cache, new_k, new_v):
    # TODO: extend the per-layer KV cache by appending new_k and new_v along the time axis.
    k_cache = kv_cache['k']
    v_cache = kv_cache['v']
    if k_cache is None:
        k_cache = new_k
        v_cache = new_v
    else:
        k_cache = np.concatenate((k_cache, new_k), axis=0)
        v_cache = np.concatenate((v_cache, new_v), axis=0)
    return {
        'k': k_cache,
        'v': v_cache
    }

# Step 14 - scaled_dot_product_attention_with_cache
import numpy as np

def scaled_dot_product_attention_with_cache(queries, kv_cache, query_offset=0):
    """Causal scaled dot-product attention of queries against a KV cache."""
    # TODO: combine score, scale, mask, softmax, and weighted value sum primitives.
    T_q, d_head = queries.shape
    keys = kv_cache['k']
    values = kv_cache['v']
    attn_scores = compute_attention_scores(queries, keys)
    attn_scores = scale_attention_scores(attn_scores, d_head)
    masked_scores = apply_causal_mask(attn_scores, query_offset)
    attn_weights = softmax_attention_weights(masked_scores)
    output = weighted_value_sum(attn_weights, values)
    return output

# Step 15 - apply_output_projection
def apply_output_projection(context, attn_params):
    # TODO: project the attention context back to model dimension using attn_params['Wo'] and attn_params['bo'].
    return linear_projection(context, attn_params['Wo'], attn_params['bo'])

# Step 16 - single_head_causal_self_attention
import numpy as np

def single_head_causal_self_attention(x, attn_params, kv_cache, query_offset=0):
    """Single-head causal self-attention with KV-cache update.

    Returns (out, kv_cache) where out has shape (T, d_model).
    """
    # TODO: project to q,k,v; update kv_cache; run causal attention; output projection.
    q, k, v = project_qkv(x, attn_params)
    kv_cache = append_kv_cache(kv_cache, k, v)
    context = scaled_dot_product_attention_with_cache(q, kv_cache, query_offset)
    output = apply_output_projection(context, attn_params)
    return output, kv_cache

# Step 17 - ffn_first_layer_gelu
def gelu(z):
    inner = z + 0.044715 * (z ** 3)
    inner = np.sqrt(2/np.pi) * inner
    return 0.5 * z * (1 + np.tanh(inner))



def ffn_first_layer_gelu(x, ffn_params):
    # TODO: apply the first linear layer of the FFN and then a GELU activation.
    w1 = ffn_params['W1']
    b1 = ffn_params['b1'] if 'b1' in ffn_params else None
    linear_output = linear_projection(x, w1, b1)
    return gelu(linear_output)

# Step 18 - ffn_second_layer
def ffn_second_layer(h, ffn_params):
    # TODO: apply the second FFN linear layer mapping (T, d_ff) back to (T, d_model).
    return linear_projection(h, ffn_params['W2'], ffn_params['b2'])

# Step 19 - position_wise_feed_forward
def position_wise_feed_forward(x, ffn_params):
    # TODO: apply the FFN as two linear layers with a GELU between them
    x = ffn_first_layer_gelu(x, ffn_params)
    return ffn_second_layer(x, ffn_params)

# Step 20 - compute_mean_variance
import numpy as np

def compute_mean_variance(x, eps=1e-5):
    """Compute per-feature mean and variance along the last axis of x."""
    # TODO: return (mean, var) reduced along the last axis with that axis kept
    x_mean = np.mean(x, axis=-1, keepdims=True)
    x_var = np.var(x, axis=-1, keepdims=True)
    return (x_mean, x_var)

# Step 21 - layer_norm_apply
import numpy as np

def layer_norm_apply(x, ln_params, eps=1e-5):
    """Normalize x over its last axis and apply gamma, beta."""
    # TODO: normalize x across its last axis and apply the affine transform.
    x_mean, x_var = compute_mean_variance(x, eps)
    x = (x - x_mean) / np.sqrt(x_var + eps)
    return x * ln_params['gamma'] + ln_params['beta']

# Step 22 - residual_add_and_norm
import numpy as np

def residual_add_and_norm(x, sublayer_output, ln_params, eps=1e-5):
    # TODO: combine the residual connection with layer normalization over the feature axis.
    x = x + sublayer_output
    return layer_norm_apply(x, ln_params, eps)

# Step 23 - transformer_block (not yet solved)
# TODO: implement

# Step 24 - lm_head_logits (not yet solved)
# TODO: implement

# Step 25 - greedy_next_token (not yet solved)
# TODO: implement

# Step 26 - run_prefill (not yet solved)
# TODO: implement

# Step 27 - decode_step (not yet solved)
# TODO: implement

# Step 28 - generate_with_state_log (not yet solved)
# TODO: implement

# Step 29 - hash_tensor (not yet solved)
# TODO: implement

# Step 30 - commit_decode_step (not yet solved)
# TODO: implement

# Step 31 - hash_pair (not yet solved)
# TODO: implement

# Step 32 - build_merkle_level (not yet solved)
# TODO: implement

# Step 33 - build_merkle_tree (not yet solved)
# TODO: implement

# Step 34 - merkle_root (not yet solved)
# TODO: implement

# Step 35 - merkle_inclusion_proof (not yet solved)
# TODO: implement

# Step 36 - verify_merkle_inclusion_proof (not yet solved)
# TODO: implement

# Step 37 - run_prover (not yet solved)
# TODO: implement

# Step 38 - assemble_public_transcript (not yet solved)
# TODO: implement

# Step 39 - sample_audit_positions (not yet solved)
# TODO: implement

# Step 40 - reexecute_audited_step (not yet solved)
# TODO: implement

# Step 41 - recompute_step_commitment (not yet solved)
# TODO: implement

# Step 42 - check_commitment_against_proof (not yet solved)
# TODO: implement

# Step 43 - check_token_matches_claim (not yet solved)
# TODO: implement

# Step 44 - run_spot_check_verification (not yet solved)
# TODO: implement

# Step 45 - tamper_transcript_flip_token (not yet solved)
# TODO: implement

# Step 46 - detection_probability (not yet solved)
# TODO: implement

# Step 47 - verifier_cost_fraction (not yet solved)
# TODO: implement

# Step 48 - show_tampered_transcript_rejected (not yet solved)
# TODO: implement

# Step 49 - sample_verifier_committee (not yet solved)
# TODO: implement

# Step 50 - collect_verifier_votes (not yet solved)
# TODO: implement

# Step 51 - aggregate_votes_majority (not yet solved)
# TODO: implement

# Step 52 - reward_honest_participants (not yet solved)
# TODO: implement

# Step 53 - slash_worker (not yet solved)
# TODO: implement

# Step 54 - assign_dual_role (not yet solved)
# TODO: implement

# Step 55 - run_honest_round (not yet solved)
# TODO: implement

# Step 56 - run_malicious_round (not yet solved)
# TODO: implement

# Step 57 - report_end_to_end_verification_cost (not yet solved)
# TODO: implement

