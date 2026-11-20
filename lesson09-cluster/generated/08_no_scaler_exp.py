# -*- coding: utf-8 -*-
"""
实践指南九 · 故意实验：不做标准化直接聚类
balance（十万级）会碾压一切——预期分群变成"按余额切四段"。
对比标准化/不标准化两种结果的画像表，亲眼看到量纲陷阱。
红线2对照实验：本实验故意违反"距离类模型必须先标准化"，仅作教学演示。
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------- 中文显示设置（Windows 可用字体） ----------
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------- 读取数据 ----------
df = pd.read_csv("data/bank_customers.csv")
FEATS = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]

# 先摆出量纲差距的证据
print("原始四列量纲对比（max 值）：")
for f in FEATS:
    print(f"  {f}: max={df[f].max():,.2f}")
print("余额是逾期次数的 10 万倍量级——欧氏距离里谁数字大谁说了算。\n")

# ---------- 实验 A：不标准化直接聚类（故意翻车） ----------
km_raw = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_raw"] = km_raw.fit_predict(df[FEATS])

# ---------- 实验 B：标准化后聚类（正确做法） ----------
X = StandardScaler().fit_transform(df[FEATS])
km_std = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_std"] = km_std.fit_predict(X)

# ---------- 两种结果的画像表对比 ----------
agg = dict(
    人数=("customer_id", "count"),
    平均余额元=("balance", "mean"),
    平均月消费元=("avg_monthly_spend", "mean"),
    平均产品数=("product_count", "mean"),
    平均逾期=("overdue_count", "mean"),
)
prof_raw = df.groupby("cluster_raw").agg(**agg).round(2)
prof_std = df.groupby("cluster_std").agg(**agg).round(2)

print("【不标准化】画像表（预期：余额从低到高切成四段，其他特征被抹平）：")
print(prof_raw.to_string())
print("\n【标准化后】画像表（预期：余额接近，消费/产品/逾期拉开区分）：")
print(prof_std.to_string())

# 两个结果是否一致？
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(df["cluster_raw"], df["cluster_std"])
print(f"\n两种分法的调整兰德指数 ARI={ari:.3f}（1=完全一致，0=随机，<0.5 即大幅不同）")

# ---------- 可视化：同一坐标、两种着色 ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 7), sharey=True)
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

for ax, col, title in [
    (axes[0], "cluster_raw", "不标准化：余额（10万级）主导距离\n≈ 按余额切四段"),
    (axes[1], "cluster_std", "标准化后：四特征等权\n消费/产品/逾期拉开区分"),
]:
    for c in range(4):
        sub = df[df[col] == c]
        ax.scatter(sub["balance"], sub["avg_monthly_spend"], s=30, alpha=0.7,
                   color=colors[c], edgecolors="white", linewidths=0.3,
                   label=f"簇{c}（{len(sub)}人）")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("账户余额（元）", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
axes[0].set_ylabel("月均消费（元）", fontsize=11)
fig.suptitle("故意实验：标准化前后聚类结果对比（k=4，同一批 200 位客户）", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("outputs/11_no_scaler_compare.png", dpi=150)
print("\n已保存 outputs/11_no_scaler_compare.png")
