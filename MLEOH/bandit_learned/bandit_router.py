import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_trace(trace_path: str) -> pd.DataFrame:
    df = pd.read_csv(trace_path)
    # normalize
    df["chosen_model"] = df["chosen_model"].astype(str).str.lower().str.strip()
    df["operator"] = df["operator"].astype(str).str.upper().str.strip()

    # 兜底
    df.loc[~df["chosen_model"].isin(["deepseek", "qwen"]), "chosen_model"] = "unknown"
    df.loc[df["operator"].isin(["NONE", "NAN", ""]), "operator"] = "UNKNOWN"
    return df


def plot_operator_stacked(df: pd.DataFrame, out_path: str):
    """
    图A：按 operator 展示选择比例（堆叠柱状图）
    """
    # count table
    g = df.groupby(["operator", "chosen_model"]).size().reset_index(name="count")
    pivot = g.pivot_table(index="operator", columns="chosen_model", values="count", fill_value=0)

    # 保证列存在
    for col in ["deepseek", "qwen"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["deepseek", "qwen"]]

    # proportion
    prop = pivot.div(pivot.sum(axis=1), axis=0)

    plt.figure(figsize=(8, 4))
    bottom = np.zeros(len(prop))
    for col in prop.columns:
        plt.bar(prop.index, prop[col].values, bottom=bottom, label=col)
        bottom += prop[col].values

    plt.ylim(0, 1)
    plt.ylabel("Choice Proportion")
    plt.title("Bandit Router: Choice Distribution by Operator")
    plt.xticks(rotation=20)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"[OK] saved: {out_path}")


def plot_cumulative_curve(df: pd.DataFrame, out_path: str):
    """
    图B：累计比例曲线（随 step 变化）
    y轴：截至当前 step，deepseek 的累计选择比例
    """
    df2 = df.sort_values("step").reset_index(drop=True)
    is_ds = (df2["chosen_model"] == "deepseek").astype(int).values
    cum_ds = np.cumsum(is_ds)
    steps = np.arange(1, len(df2) + 1)
    cum_prop = cum_ds / steps

    plt.figure(figsize=(8, 4))
    plt.plot(steps, cum_prop)
    plt.ylim(0, 1)
    plt.xlabel("Routing Step")
    plt.ylabel("Cumulative P(choose deepseek)")
    plt.title("Bandit Router: Cumulative Choice Preference Over Time")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"[OK] saved: {out_path}")


def plot_heatmap(df: pd.DataFrame, out_path: str):
    """
    图C：Operator x Model 热力图（比例）
    """
    g = df.groupby(["operator", "chosen_model"]).size().reset_index(name="count")
    pivot = g.pivot_table(index="operator", columns="chosen_model", values="count", fill_value=0)

    for col in ["deepseek", "qwen"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["deepseek", "qwen"]]
    prop = pivot.div(pivot.sum(axis=1), axis=0)

    mat = prop.values
    ops = prop.index.tolist()
    models = prop.columns.tolist()

    plt.figure(figsize=(6, 4))
    plt.imshow(mat, aspect="auto")  # 不指定颜色，默认即可
    plt.xticks(range(len(models)), models)
    plt.yticks(range(len(ops)), ops)
    plt.colorbar(label="Choice Proportion")
    plt.title("Bandit Router: Operator-Model Preference Heatmap")

    # 标注数值（更像论文）
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"[OK] saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=str, required=True, help="Path to router_trace.csv")
    parser.add_argument("--out_dir", type=str, default=".", help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_trace(args.trace)

    plot_operator_stacked(df, os.path.join(args.out_dir, "fig_bandit_choice_by_operator.png"))
    plot_cumulative_curve(df, os.path.join(args.out_dir, "fig_bandit_choice_cumulative.png"))
    plot_heatmap(df, os.path.join(args.out_dir, "fig_bandit_choice_heatmap.png"))


if __name__ == "__main__":
    main()
