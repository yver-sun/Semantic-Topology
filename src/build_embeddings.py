import os
import re
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from src.stopwords_en import STOPWORDS


WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")


def iter_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text)
    # naive sentence split
    sentences = re.split(r"(?<=[\.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def merge_wordpieces(tokens: List[str]) -> List[Tuple[str, List[int]]]:
    words: List[Tuple[str, List[int]]] = []
    cur = []
    idxs = []
    for i, t in enumerate(tokens):
        if t.startswith("##"):
            cur.append(t[2:])
            idxs.append(i)
        else:
            if cur:
                words.append(("".join(cur), idxs))
            cur = [t]
            idxs = [i]
    if cur:
        words.append(("".join(cur), idxs))
    return words


def is_content_word(w: str) -> bool:
    lw = w.lower()
    if lw in STOPWORDS:
        return False
    if len(lw) <= 2:
        return False
    if not WORD_RE.fullmatch(lw):
        return False
    return True


def embed_text_file(txt_path: str, out_dir: str, freq_min: int = 5) -> str:
    os.makedirs(out_dir, exist_ok=True)
    model_name = os.environ.get("MODEL_NAME", "bert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    sentences = iter_sentences(text)
    max_s = os.environ.get("MAX_SENTENCES")
    if max_s:
        try:
            sentences = sentences[:int(max_s)]
        except Exception:
            pass
    freq: Dict[str, int] = {}
    occurrences: Dict[str, List[np.ndarray]] = {}

    for s in sentences:
        if not s:
            continue
        enc = tokenizer(s, return_tensors='pt', truncation=True, max_length=256)
        with torch.no_grad():
            out = model(**enc)
        hidden = out.last_hidden_state.squeeze(0)  # [L, 768]
        pieces = tokenizer.convert_ids_to_tokens(enc['input_ids'].squeeze(0).tolist())
        words = merge_wordpieces(pieces)
        for word, idxs in words:
            if not is_content_word(word):
                continue
            vec = hidden[idxs].mean(dim=0).cpu().numpy()
            lw = word.lower()
            freq[lw] = freq.get(lw, 0) + 1
            occurrences.setdefault(lw, []).append(vec)

    # apply frequency threshold (env override supported)
    try:
        freq_min = int(os.environ.get("FREQ_MIN", str(freq_min)))
    except Exception:
        pass
    selected = {w: occ for w, occ in occurrences.items() if freq.get(w, 0) >= freq_min and len(occ) >= freq_min}
    labels: List[str] = []
    vectors: List[np.ndarray] = []
    for w, occs in selected.items():
        for v in occs:
            labels.append(w)
            vectors.append(v)

    X = np.stack(vectors, axis=0) if vectors else np.empty((0, 768), dtype=np.float32)
    labels_arr = np.array(labels, dtype=object)
    out_path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(txt_path))[0]}_embeddings.npz")
    np.savez(out_path, X=X, labels=labels_arr)
    return out_path


if __name__ == "__main__":
    root = os.getcwd()
    txt_dir = os.path.join(root, "artifacts", "texts")
    out_dir = os.path.join(root, "artifacts", "embeddings")
    for name in os.listdir(txt_dir):
        if name.endswith('.txt'):
            p = os.path.join(txt_dir, name)
            out = embed_text_file(p, out_dir, freq_min=5)
            print(name, "->", out)
