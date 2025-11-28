import os
from src.extract_text import extract_all
from src.build_embeddings import embed_text_file
from src.tda_analysis import analyze
from src.mapper_visualization import visualize


def main():
    root = os.getcwd()
    texts_dir = os.path.join(root, 'artifacts', 'texts')
    embeds_dir = os.path.join(root, 'artifacts', 'embeddings')
    tda_dir = os.path.join(root, 'artifacts', 'tda')
    mapper_dir = os.path.join(root, 'artifacts', 'mapper')
    results_dir = os.path.join(root, 'artifacts', 'results')
    os.makedirs(results_dir, exist_ok=True)
    md_path = os.path.join(results_dir, 'analysis.md')
    # parameters from environment
    try:
        freq_min_env = int(os.environ.get('FREQ_MIN', '5'))
    except Exception:
        freq_min_env = 5
    try:
        k_landmarks_env = int(os.environ.get('K_LANDMARKS', '512'))
    except Exception:
        k_landmarks_env = 512

    extracted = extract_all(root, texts_dir)
    for name, txt_path in extracted:
        emb_path = embed_text_file(txt_path, embeds_dir, freq_min=freq_min_env)
        tda_path, cycle_words, metrics = analyze(emb_path, tda_dir, k_landmarks=k_landmarks_env)
        html_path, summary_path = visualize(emb_path, mapper_dir)
        print(f"{name} -> embeddings: {emb_path}")
        print(f"{name} -> β1 result: {tda_path}")
        print(f"top cycle words (sample): {', '.join(cycle_words[:40])}")
        try:
            with open(md_path, 'a', encoding='utf-8') as md:
                md.write(f"\n## {name} 分析结果\n")
                md.write(f"- 嵌入文件：{emb_path}\n")
                md.write(f"- β1 结果文件：{tda_path}\n")
                b, d, p = metrics
                md.write(f"- β1 条码：birth={b:.6f}, death={d:.6f}, persistence={p:.6f}\n")
                if cycle_words:
                    md.write(f"- 环边界词（样本≤40）：{', '.join(cycle_words[:40])}\n")
                else:
                    md.write(f"- 当前尺度未检出显著 β1 环或词集为空。\n")
                if html_path:
                    md.write(f"- Mapper 骨架：{html_path}\n")
                if summary_path:
                    md.write(f"- Mapper 摘要：{summary_path}\n")
        except Exception:
            pass


if __name__ == '__main__':
    main()
