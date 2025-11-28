import os
import numpy as np


def main():
    root = os.getcwd()
    emb_dir = os.path.join(root, 'artifacts', 'embeddings')
    out = os.path.join(root, 'artifacts', 'results', 'emb_shapes.txt')
    lines = []
    for name in os.listdir(emb_dir):
        if not name.endswith('_embeddings.npz'):
            continue
        fp = os.path.join(emb_dir, name)
        try:
            d = np.load(fp, allow_pickle=True)
            X = d['X']
            labels = d['labels']
            lines.append(f"{name}\t{X.shape}\t{len(labels)}")
        except Exception as e:
            lines.append(f"{name}\tERROR\t{e}")
    os.makedirs(os.path.join(root, 'artifacts', 'results'), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


if __name__ == '__main__':
    main()
