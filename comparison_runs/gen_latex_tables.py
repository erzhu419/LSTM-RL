#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_latex_tables.py  ——  从 cross_sigma_all_results.csv / cross_sigma_best_results.csv
生成 LaTeX 表格（列内最优自动加粗）。支持多种列名别名。
用法：
  python gen_latex_tables.py --all cross_sigma_all_results.csv --best cross_sigma_best_results.csv
可选：
  --out-all tables_crosssigma.tex
  --out-best table_best.tex
"""
import argparse, os, sys
import pandas as pd
import numpy as np

COL_ALIASES = {
    'algo': ['algorithm','algo','method'],
    'embed': ['embedding','embedding_mode','embed','repr','representation'],
    'sigma_train': ['sigma_train','train_sigma','best_train_sigma','sigma_tr','sigma (train)','train-sigma'],
    'sigma_test':  ['sigma_test','test_sigma','eval_sigma','sigma_te','sigma (test)','test-sigma'],
    'mean': ['mean','reward_mean','mean_reward','best_mean_reward','avg','score','return_mean'],
    'std':  ['std','reward_std','stdev','stderr','return_std'],
}

ALGO_DISPLAY = {
    'sac_v2_bus': 'SAC',
    'ddpg_bus': 'DDPG',
    'td3_bus': 'TD3',
    'maddpg': 'MADDPG',
}

EMBED_DISPLAY = {
    'full': 'Full',
    'one_hot': 'One-hot',
    'one-hot': 'One-hot',
    'none': 'None',
}

def _normalize_cols(df: pd.DataFrame):
    lower_map = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=lower_map)
    colmap = {}
    for key, aliases in COL_ALIASES.items():
        hit = next((a for a in aliases if a in df.columns), None)
        if hit is None:
            raise ValueError(f"CSV 缺少必须列: {key}（支持的名称：{aliases}）。当前列：{list(df.columns)}")
        colmap[key] = hit
    return df, colmap

def _fmt(mean, std):
    # 将数值缩放到 K（千）级别并保留一位小数
    def fmt_k(x):
        return f"{x/1000:.1f}K" if abs(x) >= 1000 else f"{x:.1f}"

    return rf"{fmt_k(mean)} $\pm$ {fmt_k(std)}"


def _display_algo(name: str) -> str:
    key = name.strip().lower()
    return ALGO_DISPLAY.get(key, name.replace('_', ' ').title())


def _display_embed(name: str) -> str:
    key = name.strip().lower()
    return EMBED_DISPLAY.get(key, name.replace('_', ' ').title())

def build_full_table(df: pd.DataFrame, colmap):
    df = df.copy()
    df['_rowkey'] = df[colmap['algo']].astype(str) + '||' + df[colmap['embed']].astype(str)
    row_order = df['_rowkey'].drop_duplicates().tolist()
    sigma_tr_vals = sorted(df[colmap['sigma_train']].unique())
    sigma_te_vals = sorted(df[colmap['sigma_test']].unique())

    # 每列（固定 test σ 和 train σ）算最大均值以便加粗
    maxima = {}
    for te in sigma_te_vals:
        for tr in sigma_tr_vals:
            sub = df[(df[colmap['sigma_test']]==te) & (df[colmap['sigma_train']]==tr)]
            maxima[(te,tr)] = sub[colmap['mean']].max() if len(sub) else -np.inf

    # 头部
    parts = []
    parts.append(r"\begin{table*}[t]\centering")
    parts.append(r"\caption{Cross-variance evaluation (mean return $\pm$ std). Best in each column is \textbf{bold}.}\label{tab:crosssigma}")
    parts.append(r"\resizebox{\textwidth}{!}{%")
    parts.append(r"\begin{tabular}{l l " + " ".join(["c"*len(sigma_tr_vals) for _ in sigma_te_vals]).replace(" ", "") + r"}")
    parts.append(r"\toprule")
    # 一级列头
    subcols = " & ".join([rf"\multicolumn{{{len(sigma_tr_vals)}}}{{c}}{{$\sigma_{{\text{{test}}}}={te}$}}" for te in sigma_te_vals])
    parts.append(r"\multirow{2}{*}{Algorithm} & \multirow{2}{*}{Embedding} & " + subcols + r" \\")
    # cmidrule
    start = 3
    rules = []
    for _ in sigma_te_vals:
        end = start + len(sigma_tr_vals) - 1
        rules.append(rf"\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    parts.append(" ".join(rules))
    # 二级列头
    parts.append(" &  & " + " & ".join([" & ".join([f"train {tr}" for tr in sigma_tr_vals]) for _ in sigma_te_vals]) + r" \\")
    parts.append(r"\midrule")

    # 行
    for rk in row_order:
        sub = df[df['_rowkey']==rk]
        algo = _display_algo(str(sub[colmap['algo']].iloc[0]))
        emb  = _display_embed(str(sub[colmap['embed']].iloc[0]))
        cells = []
        for te in sigma_te_vals:
            for tr in sigma_tr_vals:
                hit = sub[(sub[colmap['sigma_test']]==te) & (sub[colmap['sigma_train']]==tr)]
                if len(hit)==0:
                    cells.append("--")
                else:
                    m = float(hit[colmap['mean']].iloc[0]); s = float(hit[colmap['std']].iloc[0])
                    txt = _fmt(m, s)
                    if np.isclose(m, maxima[(te,tr)]):
                        cells.append(rf"\textbf{{{txt}}}")
                    else:
                        cells.append(txt)
        parts.append(f"{algo} & {emb} & " + " & ".join(cells) + r" \\")
    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}}")
    parts.append(r"\end{table*}")
    return "\n".join(parts)

def build_best_table(df: pd.DataFrame, colmap):
    te_vals = sorted(df[colmap['sigma_test']].unique())
    rows = []
    for te in te_vals:
        sub = df[df[colmap['sigma_test']]==te]
        if len(sub)==0: continue
        best_idx = sub[colmap['mean']].idxmax()
        rows.append(sub.loc[best_idx])
    best = pd.DataFrame(rows)

    parts = []
    parts.append(r"\begin{table}[t]\centering")
    parts.append(r"\caption{Best result per test variance ($\sigma_{\text{test}}$).}\label{tab:best}")
    parts.append(r"\begin{tabular}{l l c c}")
    parts.append(r"\toprule")
    parts.append(r"Algorithm & Embedding & train $\sigma$ & mean $\pm$ std \\")
    parts.append(r"\midrule")
    for _,r in best.iterrows():
        algo = _display_algo(str(r[colmap['algo']]))
        emb = _display_embed(str(r[colmap['embed']]))
        tr = r[colmap['sigma_train']]; m = float(r[colmap['mean']]); s = float(r[colmap['std']])
        parts.append(rf"{algo} & {emb} & {tr} & \textbf{{{_fmt(m, s)}}} \\")
    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}")
    parts.append(r"\end{table}")
    return "\n".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", dest="all_csv", default=None, help="path to cross_sigma_all_results.csv")
    ap.add_argument("--best", dest="best_csv", default=None, help="path to cross_sigma_best_results.csv")
    ap.add_argument("--out-all", dest="out_all", default="tables_crosssigma.tex", help="output tex for full table")
    ap.add_argument("--out-best", dest="out_best", default="table_best.tex", help="output tex for best table")
    args = ap.parse_args()

    if not args.all_csv and not args.best_csv:
        print("请至少提供 --all 或 --best 中的一个 CSV 路径。", file=sys.stderr)
        sys.exit(1)

    if args.all_csv:
        df_all = pd.read_csv(args.all_csv)
        df_all, cmap_all = _normalize_cols(df_all)
        tex_all = build_full_table(df_all, cmap_all)
        with open(args.out_all, "w", encoding="utf-8") as f:
            f.write(tex_all)
        print(f"[OK] 写出完整表格 -> {args.out_all}")

    if args.best_csv:
        df_best = pd.read_csv(args.best_csv)
        df_best, cmap_best = _normalize_cols(df_best)
        tex_best = build_best_table(df_best, cmap_best)
        with open(args.out_best, "w", encoding="utf-8") as f:
            f.write(tex_best)
        print(f"[OK] 写出最佳结果表 -> {args.out_best}")

if __name__ == "__main__":
    main()
