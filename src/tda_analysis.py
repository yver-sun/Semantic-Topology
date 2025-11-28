import os
from typing import Dict, List, Tuple

import numpy as np
from ripser import ripser


def choose_landmarks(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    if n == 0:
        return np.empty((0, X.shape[1])), np.array([], dtype=int)
    first = np.random.randint(0, n)
    idxs = [first]
    dists = np.linalg.norm(X - X[first], axis=1)
    for _ in range(1, k):
        next_idx = int(np.argmax(dists))
        idxs.append(next_idx)
        new_d = np.linalg.norm(X - X[next_idx], axis=1)
        dists = np.minimum(dists, new_d)
    L = X[idxs]
    return L, np.array(idxs, dtype=int)


def compute_persistence(L: np.ndarray, maxdim: int = 1) -> Dict:
    res = ripser(L, maxdim=maxdim, do_cocycles=True)
    return res


def pick_top_beta1_cycle(res: Dict) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
    dgms = res.get('dgms', [])
    cocycles = res.get('cocycles', [])
    if len(dgms) < 2:
        return np.array([]), []
    d1 = dgms[1]
    if d1.shape[0] == 0:
        return np.array([]), []
    pers = d1[:, 1] - d1[:, 0]
    top_idx = int(np.argmax(pers))
    cycle = cocycles[1][top_idx]
    edges = []
    for e in cycle:
        i, j, w = int(e[0]), int(e[1]), float(e[2])
        edges.append((i, j, w))
    return d1[top_idx], edges


def map_edges_to_words(edges: List[Tuple[int, int, float]], labels: np.ndarray, landmark_indices: np.ndarray) -> List[str]:
    words = []
    for i, j, _ in edges:
        wi = labels[landmark_indices[i]]
        wj = labels[landmark_indices[j]]
        words.extend([wi, wj])
    # unique order-preserving
    seen = set()
    ordered = []
    for w in words:
        if w not in seen:
            seen.add(w)
            ordered.append(w)
    return ordered


def analyze(npz_path: str, out_dir: str, k_landmarks: int = 512) -> Tuple[str, List[str], Tuple[float, float, float]]:
    os.makedirs(out_dir, exist_ok=True)
    data = np.load(npz_path, allow_pickle=True)
    X: np.ndarray = data['X']
    labels: np.ndarray = data['labels']
    if X.shape[0] == 0:
        return "", [], (0.0, 0.0, 0.0)
    k = min(k_landmarks, max(8, X.shape[0]))
    L, landmark_indices = choose_landmarks(X, k)
    res = compute_persistence(L, maxdim=1)
    top_bar, edges = pick_top_beta1_cycle(res)
    words = map_edges_to_words(edges, labels, landmark_indices)
    out_path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(npz_path))[0]}_beta1.npy")
    np.save(out_path, {
        'diagram': res.get('dgms'),
        'top_bar': top_bar,
        'edges': edges,
        'landmark_indices': landmark_indices,
        'words_on_cycle': np.array(words, dtype=object)
    })
    birth = float(top_bar[0]) if top_bar.size else 0.0
    death = float(top_bar[1]) if top_bar.size else 0.0
    persistence = death - birth if top_bar.size else 0.0
    return out_path, words, (birth, death, persistence)


if __name__ == "__main__":
    root = os.getcwd()
    emb_dir = os.path.join(root, "artifacts", "embeddings")
    out_dir = os.path.join(root, "artifacts", "tda")
    for name in os.listdir(emb_dir):
        if name.endswith('_embeddings.npz'):
            p = os.path.join(emb_dir, name)
            out, words = analyze(p, out_dir)
            print(name, "->", out)
            print("cycle words:", ", ".join(words[:32]))
