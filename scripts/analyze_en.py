import os
from src.tda_analysis import analyze
from src.mapper_visualization import visualize


def main():
    root = os.getcwd()
    emb_dir = os.path.join(root, 'artifacts', 'embeddings')
    tda_dir = os.path.join(root, 'artifacts', 'tda')
    map_dir = os.path.join(root, 'artifacts', 'mapper')
    res_md = os.path.join(root, 'artifacts', 'results', 'analysis.md')
    os.makedirs(tda_dir, exist_ok=True)
    os.makedirs(map_dir, exist_ok=True)
    targets = [
        '2010 Point Omega (Delillo, Don [Delillo, Don]) (Z-Library)_embeddings.npz',
        '2016 Zero K  a novel (DeLillo, Don, author) (Z-Library)_embeddings.npz',
        '2020 The Silence (Don DeLillo) (Z-Library)_embeddings.npz',
    ]
    for name in targets:
        npz = os.path.join(emb_dir, name)
        if not os.path.exists(npz):
            continue
        status_line = []
        try:
            tda_path, cycle_words, metrics = analyze(npz, tda_dir, k_landmarks=256)
            status_line.append("ANALYZE_OK")
        except Exception as e:
            tda_path, cycle_words, metrics = "", [], (0.0, 0.0, 0.0)
            status_line.append(f"ANALYZE_ERR:{e}")
        try:
            html_path, summary_path = visualize(npz, map_dir)
            status_line.append("VIS_OK")
        except Exception as e:
            html_path, summary_path = "", ""
            status_line.append(f"VIS_ERR:{e}")
        base = name.replace('_embeddings.npz', '')
        with open(res_md, 'a', encoding='utf-8') as md:
            md.write(f"\n## {base} 分析结果\n")
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
        try:
            with open(os.path.join(root, 'artifacts', 'results', 'tda_status.txt'), 'a', encoding='utf-8') as st:
                st.write(f"{name}\t{'|'.join(status_line)}\n")
        except Exception:
            pass


if __name__ == '__main__':
    main()
