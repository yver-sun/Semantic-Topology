import os
from src.tda_analysis import analyze
from src.mapper_visualization import visualize


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def main():
    root = os.getcwd()
    emb_dir = os.path.join(root, 'artifacts', 'embeddings')
    tda_dir = os.path.join(root, 'artifacts', 'tda')
    map_dir = os.path.join(root, 'artifacts', 'mapper')
    res_md = os.path.join(root, 'artifacts', 'results', 'analysis.md')
    os.makedirs(tda_dir, exist_ok=True)
    os.makedirs(map_dir, exist_ok=True)
    k_landmarks = env_int('K_LANDMARKS', 256)

    for name in os.listdir(emb_dir):
        if not name.endswith('_embeddings.npz'):
            continue
        npz = os.path.join(emb_dir, name)
        try:
            tda_path, cycle_words, metrics = analyze(npz, tda_dir, k_landmarks=k_landmarks)
        except Exception:
            tda_path, cycle_words, metrics = "", [], (0.0, 0.0, 0.0)
        try:
            html_path, summary_path = visualize(npz, map_dir)
        except Exception:
            html_path, summary_path = "", ""
        try:
            with open(res_md, 'a', encoding='utf-8') as md:
                md.write(f"\n## {name.replace('_embeddings.npz','')} 分析结果\n")
                md.write(f"- 嵌入文件：{npz}\n")
                md.write(f"- β1 结果文件：{tda_path}\n")
                b, d, p = metrics
                md.write(f"- β1 条码：birth={b:.6f}, death={d:.6f}, persistence={p:.6f}\n")
                if cycle_words:
                    md.write(f"- 环边界词（样本≤40）：{', '.join(cycle_words[:40])}\n")
                else:
                    md.write("- 当前尺度未检出显著 β1 环或词集为空。\n")
                if html_path:
                    md.write(f"- Mapper 骨架：{html_path}\n")
                if summary_path:
                    md.write(f"- Mapper 摘要：{summary_path}\n")
        except Exception:
            pass


if __name__ == '__main__':
    main()
