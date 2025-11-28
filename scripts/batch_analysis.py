import os
from src.build_embeddings import embed_text_file
from src.tda_analysis import analyze
from src.mapper_visualization import visualize


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def main():
    root = os.getcwd()
    texts = [
        os.path.join(root, 'artifacts', 'texts', '2010 Point Omega (Delillo, Don [Delillo, Don]) (Z-Library).txt'),
        os.path.join(root, 'artifacts', 'texts', '2016 Zero K  a novel (DeLillo, Don, author) (Z-Library).txt'),
        os.path.join(root, 'artifacts', 'texts', '2020 The Silence (Don DeLillo) (Z-Library).txt'),
    ]
    emb_dir = os.path.join(root, 'artifacts', 'embeddings')
    tda_dir = os.path.join(root, 'artifacts', 'tda')
    map_dir = os.path.join(root, 'artifacts', 'mapper')
    res_md = os.path.join(root, 'artifacts', 'results', 'analysis.md')
    os.makedirs(emb_dir, exist_ok=True)
    os.makedirs(tda_dir, exist_ok=True)
    os.makedirs(map_dir, exist_ok=True)

    freq_min = env_int('FREQ_MIN', 1)
    k_landmarks = env_int('K_LANDMARKS', 256)

    for txt in texts:
        if not os.path.exists(txt):
            continue
        try:
            emb = embed_text_file(txt, emb_dir, freq_min)
        except Exception:
            continue
        try:
            tda_path, cycle_words, metrics = analyze(emb, tda_dir, k_landmarks=k_landmarks)
        except Exception:
            tda_path, cycle_words, metrics = "", [], (0.0, 0.0, 0.0)
        try:
            html_path, summary_path = visualize(emb, map_dir)
        except Exception:
            html_path, summary_path = "", ""
        name = os.path.basename(txt)
        try:
            with open(res_md, 'a', encoding='utf-8') as md:
                md.write(f"\n## {name} 分析结果\n")
                md.write(f"- 嵌入文件：{emb}\n")
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
