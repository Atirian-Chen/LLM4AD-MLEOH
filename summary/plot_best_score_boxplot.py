import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def load_best_scores(csv_path: str) -> list[float]:
    df = pd.read_csv(csv_path)
    if "best_score" not in df.columns:
        raise ValueError(f"[{csv_path}] missing column: best_score")
    # 只取 ok=True 的行（保险）
    if "ok" in df.columns:
        df = df[df["ok"] == True]
    return df["best_score"].astype(float).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir",
        type=str,
        default=".",
        help="Directory that contains summary csv files.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="fig_best_score_boxplot.png",
        help="Output figure filename.",
    )
    args = parser.parse_args()

    # ✅ 你现在这些文件名（按你上传的来）
    # 你可以按需要删掉 / 改名字
    method_files = {
        "singleDS": "summary_singleDS.csv",
        "singleQwen": "summary_singleQwen.csv",
        "random": "summary_random.csv",
        "ruleA": "summary_ruleA.csv",
        "ruleB": "summary_ruleB.csv",
        "bandit_train": "summary_bandit_train.csv",
        # 这里默认用 seed0-4 的 bandit_test
        "bandit_test": "summary_b.csv",
    }

    data = []
    labels = []

    for method, fname in method_files.items():
        path = os.path.join(args.in_dir, fname)
        if not os.path.exists(path):
            print(f"[WARN] missing file: {path}, skip.")
            continue

        scores = load_best_scores(path)
        if len(scores) == 0:
            print(f"[WARN] empty scores: {path}, skip.")
            continue

        data.append(scores)
        labels.append(method)

    if len(data) == 0:
        raise RuntimeError("No valid summary csv loaded. Check file paths.")

    # ✅ 画箱线图（boxplot）
    plt.figure(figsize=(10, 5))
    plt.boxplot(
        data,
        labels=labels,
        showmeans=True,     # 显示均值点
        meanline=False,
    )
    plt.xticks(rotation=20)
    plt.ylabel("best_score (higher is better; closer to 0 is better)")
    plt.title("EoH-OBP: best_score distribution across methods")

    # 可选：y轴网格线让图更清晰
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"[OK] saved figure to: {args.out}")


if __name__ == "__main__":
    main()
