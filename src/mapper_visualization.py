import os
from typing import Tuple

import numpy as np
import umap
from kmapper import KeplerMapper
from sklearn.cluster import DBSCAN


def visualize(npz_path: str, out_dir: str) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    data = np.load(npz_path, allow_pickle=True)
    X: np.ndarray = data['X']
    labels: np.ndarray = data['labels']
    if X.shape[0] == 0:
        return "", ""
    mapper = KeplerMapper()
    lens = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean').fit_transform(X)
    graph = mapper.map(lens, X, clusterer=DBSCAN(eps=0.5, min_samples=10))
    html_path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(npz_path))[0]}_mapper.html")
    mapper.visualize(graph, path_html=html_path, title=os.path.basename(npz_path))
    # also save a simple summary
    summary_path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(npz_path))[0]}_mapper_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"nodes: {len(graph['nodes'])}\n")
        f.write(f"links: {len(graph['links'])}\n")
    return html_path, summary_path


if __name__ == "__main__":
    root = os.getcwd()
    emb_dir = os.path.join(root, "artifacts", "embeddings")
    out_dir = os.path.join(root, "artifacts", "mapper")
    for name in os.listdir(emb_dir):
        if name.endswith('_embeddings.npz'):
            p = os.path.join(emb_dir, name)
            html, summary = visualize(p, out_dir)
            print(name, "->", html)
