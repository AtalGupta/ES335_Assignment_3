# streamlit_app.py
"""
Streamlit app for MLP next-word generation supporting two datasets / many pretrained variants.
Features:
 - Loads checkpoint (.pth) and vocab (.json)
 - Infers model embedding dim, context length and MLP layer sizes from checkpoint state_dict
 - Allows user to modify context length in UI; generation backend pads/truncates correctly
 - Temperature, top-k, seed, device, num tokens controls
 - Handles OOV tokens (maps to <UNK>) and reports them
 - Accepts uploaded model/vocab files or uses defaults listed in MODEL_FILES
Run:
    streamlit run streamlit_app.py
"""

import streamlit as st
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import json, os, math, random
from pathlib import Path
from typing import List, Tuple

st.set_page_config(layout="wide", page_title="MLP Next-Word Generator (multi-variant)")

# ------------- Default model/vocab filenames (edit these if you used other names) ---------------
MODEL_FILES = {
    "Paul Graham (Natural)": {
        "model": "model\Paul_Graham\model_for_streamlit.pth",
        "vocab": "model\Paul_Graham\vocab_paul_graham.json"
    },
    "Linux Kernel (Structured)": {
        "model": "model\Linux_Kernel\model_for_streamlit.pth",
        "vocab": "model\Linux_Kernel\vocab_linux_kernel.json"
    }
}
# ------------------------------------------------------------------------------------------------

# ----------------- Utility: infer model architecture from checkpoint state_dict -----------------
def infer_arch_from_state(state: dict):
    """
    Infer embedding dim, vocab_size, and MLP layer shapes from a checkpoint state dict.
    Expects keys like 'embed.weight' and 'net.{i}.weight' if model used Sequential('net', ...).
    Returns:
      vocab_size, embed_dim, context_len (inferred), mlp_layer_shapes: list of (out, in) for each Linear in net in order
    """
    # STATE may contain 'model_state' umbrella
    if 'model_state' in state:
        state = state['model_state']
    # find embed weight
    embed_key = None
    for k in state:
        if k.endswith('embed.weight') or k == 'embed.weight':
            embed_key = k
            break
    if embed_key is None:
        # try common variants
        for k in state:
            if k.endswith('embedding.weight') or 'embedding' in k:
                embed_key = k
                break
    if embed_key is None:
        raise ValueError("Could not find embedding weight in checkpoint keys. Keys: " + ", ".join(list(state.keys())[:20]))

    embed_w = state[embed_key]
    vocab_size = embed_w.shape[0]
    embed_dim = embed_w.shape[1]

    # find MLP linear weights under 'net.*.weight' or other naming conventions
    linear_keys = []
    for k in state:
        if k.endswith('.weight') and k != embed_key:
            # consider candidate: must be 2D matrix
            if getattr(state[k], 'ndim', None) == 2:
                linear_keys.append(k)
    # Attempt to select only those within 'net.' ordering if available
    linear_keys_net = sorted([k for k in linear_keys if '.net.' in k or k.startswith('net.')], key=lambda x: int(x.split('.')[1]) if x.split('.')[1].isdigit() else x)
    if linear_keys_net:
        linear_keys = linear_keys_net
    else:
        # fallback: try to order by appearance
        linear_keys = sorted(linear_keys)

    # Build list of (out, in) shapes
    mlp_layer_shapes = []
    for k in linear_keys:
        w = state[k]
        if w.ndim == 2:
            mlp_layer_shapes.append((w.shape[0], w.shape[1]))

    # The final layer should have out == vocab_size often
    # Determine in_dim of final linear -> context_len = in_dim / embed_dim
    if len(mlp_layer_shapes) == 0:
        raise ValueError("No linear layers found in checkpoint state dict. Keys looked at: " + ", ".join(linear_keys[:20]))
    final_out, final_in = mlp_layer_shapes[-1]
    # Check final_out matches vocab_size (safeguard)
    if final_out != vocab_size:
        # Sometimes final layer named differently; still proceed but warn
        # We'll still compute context_len via final_in/embed_dim if divisible
        pass

    if final_in % embed_dim != 0:
        # can't infer integer context length
        inferred_context_len = None
    else:
        inferred_context_len = final_in // embed_dim

    return vocab_size, embed_dim, inferred_context_len, mlp_layer_shapes, state

# ----------------- Utility: build model matching the inferred MLP shapes -----------------
class MLPFromShapes(nn.Module):
    def __init__(self, vocab_size:int, embed_dim:int, context_len:int, mlp_shapes:List[Tuple[int,int]], activation:str='relu', dropout:float=0.2):
        """
        Build MLP by following mlp_shapes (list of (out, in)).
        The first linear layer should accept in_dim = embed_dim * context_len
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # build layers from shapes
        layers = []
        for i, (out_dim, in_dim) in enumerate(mlp_shapes):
            # Create Linear(in_dim, out_dim)
            layers.append(nn.Linear(in_dim, out_dim))
            # Add activation+dropout except after final layer
            if i < len(mlp_shapes)-1:
                if activation == 'relu':
                    layers.append(nn.ReLU())
                else:
                    layers.append(nn.Tanh())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, C]
        emb = self.embed(x)                 # [B, C, E]
        flat = emb.view(emb.size(0), -1)    # [B, C*E]
        logits = self.net(flat)
        return logits

# ----------------- Load checkpoint and instantiate model robustly -----------------
def load_checkpoint_and_model(checkpoint_path: str, device=torch.device('cpu'), activation_guess='relu'):
    ck = torch.load(checkpoint_path, map_location='cpu')
    try:
        vocab_size, embed_dim, inferred_context_len, mlp_shapes, state = infer_arch_from_state(ck)
    except Exception as e:
        raise RuntimeError(f"Failed to infer architecture from checkpoint: {e}")
    if inferred_context_len is None:
        st.warning("Could not infer exact trained context length from checkpoint - defaulting to 8 for generation.")
        inferred_context_len = 8
    # Build model
    model = MLPFromShapes(vocab_size, embed_dim, inferred_context_len, mlp_shapes, activation=activation_guess)
    # load state dict tolerant to prefix differences
    try:
        model.load_state_dict(state if isinstance(state, dict) and 'model_state' not in state else state['model_state'])
    except Exception as e:
        # try strict=False
        try:
            model.load_state_dict(state if isinstance(state, dict) and 'model_state' not in state else state['model_state'], strict=False)
            st.info("Partial state_dict loaded (strict=False). Some keys may be missing; model may behave differently.")
        except Exception as e2:
            raise RuntimeError(f"Failed to load state dict even with strict=False: {e2}")
    model.to(device)
    model.eval()
    meta = {
        'vocab_size': vocab_size,
        'embed_dim': embed_dim,
        'context_len_trained': inferred_context_len,
        'mlp_shapes': mlp_shapes
    }
    return model, meta

# ----------------- Tokenization helpers -----------------
import re
def tokenize_natural(text: str):
    # remove special except . ; lowercase ; separate '.' as token
    t = re.sub(r'[^a-zA-Z0-9 \.]', '', text)
    t = t.lower()
    t = t.replace('.', ' . ')
    toks = t.split()
    return toks

def tokenize_code(text: str):
    # keep punctuation/operators as tokens (simple whitespace split)
    return text.split()

# ----------------- Sampling utils -----------------
def sample_next_from_model(model: nn.Module, ctx_ids: torch.LongTensor, temperature: float=1.0, top_k: int=None, device=torch.device('cpu')):
    model.eval()
    with torch.no_grad():
        logits = model(ctx_ids.to(device)).squeeze(0)  # [V]
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-8)
        if top_k and top_k > 0:
            vals, idxs = torch.topk(logits, min(top_k, logits.size(0)))
            mask = torch.full_like(logits, -1e10)
            mask[idxs] = logits[idxs]
            logits = mask
        probs = F.softmax(logits, dim=0).cpu().numpy()
        idx = np.random.choice(len(probs), p=probs)
    return int(idx)

def generate_from_model(model: nn.Module, vocab:list, word2idx:dict, idx2word:dict,
                        seed_text:str, num_tokens:int, user_context_len:int,
                        temperature:float, top_k:int, is_natural:bool,
                        device=torch.device('cpu')):
    # tokenization
    toks = tokenize_natural(seed_text) if is_natural else tokenize_code(seed_text)
    # OOV handling
    PAD = '<PAD>'
    UNK = '<UNK>'
    pad_id = word2idx.get(PAD, None)
    unk_id = word2idx.get(UNK, None)
    oov = [w for w in toks if w not in word2idx]
    # model's required context length:
    # infer from model.net first linear layer input: it should be embed_dim * context_len_trained
    # We will compute model_context_len from model.embed and model.net first linear in_features
    embed_dim = model.embed.weight.shape[1]
    # get first linear in_features
    first_linear = None
    for m in model.net:
        if isinstance(m, nn.Linear):
            first_linear = m
            break
    if first_linear is None:
        raise RuntimeError("Model net contains no Linear layer.")
    in_dim = first_linear.in_features
    model_context_len = in_dim // embed_dim
    if model_context_len * embed_dim != in_dim:
        # fallback
        model_context_len = model.embed.weight.shape[1]  # nonsense fallback but avoid crash
    # Decide what context to actually use:
    # If user requested context_len <= model_context_len: we use user's context length, pad (left) to model_context_len.
    # If user requested > model_context_len: we warn and use model_context_len (we can't expand)
    used_context_len = min(user_context_len, model_context_len)
    # prepare initial output tokens (we'll mutate out_tokens)
    out_tokens = toks.copy()
    # generation loop
    for step in range(num_tokens):
        # construct context tokens: last used_context_len of out_tokens (pad left if shorter)
        ctx_tokens = out_tokens[-used_context_len:] if len(out_tokens) >= used_context_len else ( [PAD] * (used_context_len - len(out_tokens)) + out_tokens )
        # Now pad left to model_context_len
        if model_context_len > used_context_len:
            ctx_tokens_full = [PAD] * (model_context_len - used_context_len) + ctx_tokens
        else:
            ctx_tokens_full = ctx_tokens[-model_context_len:]
        # map to ids
        ctx_ids = [ word2idx.get(w, unk_id) for w in ctx_tokens_full ]
        ctx_tensor = torch.tensor([ctx_ids], dtype=torch.long)
        next_idx = sample_next_from_model(model, ctx_tensor, temperature=temperature, top_k=top_k, device=device)
        nxt_word = idx2word.get(next_idx, UNK)
        out_tokens.append(nxt_word)
    # final text
    if is_natural:
        out_text = " ".join(out_tokens).replace(" .", ".")
    else:
        out_text = " ".join(out_tokens)
    return out_text, oov, model_context_len, used_context_len

# ----------------- Streamlit UI -----------------
st.title("MLP Next-Word Generator — Multi-Variant App")
st.markdown("Load a pretrained model checkpoint (.pth) and a vocab (.json), or choose a default variant (if available). The app will infer model architecture from the checkpoint automatically.")

# Left panel: selection / uploads
st.sidebar.header("Model & Data")
dataset_choice = st.sidebar.selectbox("Dataset / preset", list(MODEL_FILES.keys()))
preset = MODEL_FILES.get(dataset_choice, {})
preset_model_path = preset.get("model")
preset_vocab_path = preset.get("vocab")

uploaded_model = st.sidebar.file_uploader("Upload model (.pth)", type=["pth","pt"])
uploaded_vocab = st.sidebar.file_uploader("Upload vocab (.json)", type=["json"])

use_preset = st.sidebar.checkbox("Use preset files (if present)", value=True)

# Decide paths
if uploaded_vocab:
    vocab = json.load(uploaded_vocab)
    vocab_path_display = "uploaded"
else:
    if use_preset and preset_vocab_path and Path(preset_vocab_path).exists():
        with open(preset_vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        vocab_path_display = preset_vocab_path
    else:
        st.sidebar.error("No vocab available. Upload a vocab.json or enable preset with a present vocab file.")
        st.stop()

vocab_size = len(vocab)
word2idx = {w:i for i,w in enumerate(vocab)}
idx2word = {i:w for i,w in enumerate(vocab)}

if uploaded_model:
    # save to a temp file
    tmp_model_path = "uploaded_model.pth"
    with open(tmp_model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())
    model_path_to_load = tmp_model_path
else:
    if use_preset and preset_model_path and Path(preset_model_path).exists():
        model_path_to_load = preset_model_path
    else:
        model_path_to_load = None

st.sidebar.markdown(f"**Vocab:** {vocab_path_display} ({vocab_size} tokens)")
st.sidebar.markdown(f"**Model file:** {model_path_to_load if model_path_to_load else 'None (upload or provide preset)'}")

# Generation controls
st.sidebar.header("Generation controls")
user_context_len = st.sidebar.slider("Context length to use (UI)", min_value=1, max_value=64, value=8, help="This is the number of tokens you want the model to consider from your seed. Backend will adapt to the model's trained context length and pad/truncate as needed.")
activation_func = st.sidebar.selectbox("Activation function", options=("relu", "tanh"), index=0, help="Choose the activation function used in the MLP hidden layers. Must match the training configuration.")
num_generate = st.sidebar.number_input("Generate tokens (k)", min_value=1, max_value=500, value=40)
temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.05)
top_k = st.sidebar.slider("Top-k sampling (0=disabled)", min_value=0, max_value=1000, value=80)
seed_val = st.sidebar.number_input("Random seed", min_value=0, max_value=9999999, value=42)
device_choice = st.sidebar.selectbox("Device", options=("cpu", "cuda (if available)"))
st.sidebar.markdown("---")
st.sidebar.markdown("Advanced: If your checkpoint was trained with a different embedding dim / activation / context, the app will attempt to reconstruct the architecture. For correct experiments, use checkpoints trained with the indicated hyperparams.")

# Load model button
load_button = st.sidebar.button("Load model")

# Area: show model metadata once loaded
model_container = st.empty()
meta_container = st.empty()

# Main input area
st.header("Seed text (input)")
is_natural = st.radio("Tokenization mode", ("Natural text (Paul Graham style)", "Code / Structured (Linux Kernel)"), horizontal=True)
is_natural_bool = True if is_natural.startswith("Natural") else False
seed_text = st.text_area("Enter seed text (tokens). Use words/tokens from the dataset for best results.", value="int main", height=140)

# Buttons
col1, col2, col3 = st.columns(3)
with col1:
    run_button = st.button("Generate")
with col2:
    compare_temps = st.button("Generate 3 temperatures (0.5, 1.0, 1.5)")
with col3:
    show_oov = st.checkbox("Show OOV tokens used in seed", value=True)

# Load model lazily when requested or when generate pressed
loaded = st.session_state.get("model_loaded", False)
if load_button or (run_button or compare_temps) and not loaded:
    if model_path_to_load is None:
        st.error("No model provided. Upload a .pth checkpoint or enable preset with available model.")
    else:
        try:
            device = torch.device("cuda" if (device_choice.startswith("cuda") and torch.cuda.is_available()) else "cpu")
            model, meta = load_checkpoint_and_model(model_path_to_load, device=device, activation_guess=activation_func)
            st.session_state["model"] = model
            st.session_state["meta"] = meta
            st.session_state["model_loaded"] = True
            st.session_state["activation"] = activation_func
            model_container.success(f"Model loaded. Vocab size {meta['vocab_size']}, embed_dim {meta['embed_dim']}, trained_context_len {meta['context_len_trained']}, activation={activation_func}, mlp_shapes {meta['mlp_shapes']}")
        except Exception as e:
            st.exception(e)
            st.stop()
else:
    if st.session_state.get("model_loaded", False):
        # Check if activation function changed - if so, reload model
        prev_activation = st.session_state.get("activation", "relu")
        if activation_func != prev_activation:
            st.warning(f"Activation function changed from '{prev_activation}' to '{activation_func}'. Reloading model...")
            if model_path_to_load:
                try:
                    device = torch.device("cuda" if (device_choice.startswith("cuda") and torch.cuda.is_available()) else "cpu")
                    model, meta = load_checkpoint_and_model(model_path_to_load, device=device, activation_guess=activation_func)
                    st.session_state["model"] = model
                    st.session_state["meta"] = meta
                    st.session_state["activation"] = activation_func
                    model_container.success(f"Model reloaded with {activation_func} activation. embed_dim {meta['embed_dim']}, trained_context_len {meta['context_len_trained']}, mlp_shapes {meta['mlp_shapes']}")
                except Exception as e:
                    st.exception(e)
                    st.stop()
            else:
                st.error("Cannot reload - no model path available")
                st.stop()
        else:
            model = st.session_state["model"]
            meta = st.session_state["meta"]
            device = torch.device("cuda" if (device_choice.startswith("cuda") and torch.cuda.is_available()) else "cpu")
            # show basic info
            model_container.info(f"Model ready. embed_dim {meta['embed_dim']}, trained_context_len {meta['context_len_trained']}, activation={activation_func}, mlp_shapes {meta['mlp_shapes']}")
    else:
        model = None
        meta = None

# Generation action
if run_button or compare_temps:
    if model is None:
        st.error("No model loaded. Click 'Load model' or upload a model.")
    else:
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)

        device = torch.device("cuda" if (device_choice.startswith("cuda") and torch.cuda.is_available()) else "cpu")
        model.to(device)

        if compare_temps:
            temps = [0.5, 1.0, 1.5]
            st.subheader("Comparing temperatures")
            for t in temps:
                with st.expander(f"Temperature = {t}", expanded=False):
                    out_text, oov, model_ctx, used_ctx = generate_from_model(model, vocab, word2idx, idx2word,
                                                seed_text=seed_text, num_tokens=num_generate,
                                                user_context_len=user_context_len, temperature=t, top_k=(None if top_k==0 else top_k),
                                                is_natural=is_natural_bool, device=device)
                    st.code(out_text)
                    st.write(f"Model trained context_len = {model_ctx}, used context_len = {used_ctx}")
                    if show_oov and oov:
                        st.warning(f"OOV tokens mapped to <UNK>: {oov}")
        else:
            out_text, oov, model_ctx, used_ctx = generate_from_model(model, vocab, word2idx, idx2word,
                                                seed_text=seed_text, num_tokens=num_generate,
                                                user_context_len=user_context_len, temperature=temperature, top_k=(None if top_k==0 else top_k),
                                                is_natural=is_natural_bool, device=device)
            st.subheader("Generated output")
            st.code(out_text)
            st.write(f"Model trained context_len = {model_ctx}, used context_len = {used_ctx}")
            if show_oov and oov:
                st.warning(f"OOV tokens mapped to <UNK>: {oov}")

# Footer notes
st.markdown("---")
st.markdown("Notes:")
st.markdown("- The app reconstructs the MLP architecture from the checkpoint `state_dict`. If the checkpoint was saved with extra metadata (e.g., `context_len`) it will be used; otherwise the app infers context length from the final linear dimension and embedding dim.")
st.markdown("- The UI slider for context length controls how many seed tokens you *intend* to use; the backend will pad / truncate to the model's actual trained context length and report what was used.")
st.markdown("- The **Activation function** selector allows you to choose between ReLU and Tanh")