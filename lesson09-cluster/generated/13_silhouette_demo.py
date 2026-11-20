# -*- coding: utf-8 -*-
"""
独立研究③：轮廓系数 silhouette_score 是什么（5分）
用本课数据（bank_customers，四特征标准化，k=4）实际算一次轮廓系数，
并顺带扫 k=2..9 看各 k 的读数，终端输出解释。
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ---------- 中文显示设置（Windows 可用字体） ----------
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------- 读取数据并标准化 ----------
df = pd.read_csv("data/bank_customers.csv")
FEATS = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]
X = StandardScaler().fit_transform(df[FEATS])

# ---------- 是什么：先讲清楚三个量 ----------
print("""
轮廓系数是什么（教材口径）：
对每个样本 i 计算
  a(i) = i 到同簇其他点的平均距离（内聚度，越小越好）
  b(i) = i 到最近邻簇各点的平均距离（分离度，越大越好）
  s(i) = (b - a) / max(a, b)，取值 [-1, 1]
全体样本的 s(i) 平均值即 silhouette_score：越接近 1 簇越紧且分得越开，
接近 0 表示样本落在簇边界上，为负说明可能分错簇。""")

# ---------- 用代码算一次：k=4（与案例一致） ----------
km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
lab4 = km4.fit_predict(X)
s4 = silhouette_score(X, lab4)
print(f"本课案例（k=4）轮廓系数 = {s4:.3f}")

# ---------- 顺带扫 k=2..9 ----------
ks = list(range(2, 10))
scores = []
for k in ks:
    lab = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    scores.append(silhouette_score(X, lab))
print("\nk=2..9 轮廓系数对照：")
for k, s in zip(ks, scores):
    mark = "  ← 本课取值" if k == 4 else ""
    print(f"  k={k}: {s:.3f}{mark}")

best_k = ks[scores.index(max(scores))]
print(f"\n轮廓系数最高的 k={best_k}（{max(scores):.3f}），"
      f"k=4 读数 {s4:.3f} 排第 {sorted(scores, reverse=True).index(s4) + 1} 名——"
      "轮廓系数只衡量'分得开不开'，不衡量'业务上有没有用'，"
      "与肘部法/业务判断互相印证而不互相替代。")

# ---------- 可视化：轮廓系数随 k 变化 ----------
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(ks, scores, marker="o", markersize=8, linewidth=2.2, color="#4C72B0",
        label="轮廓系数 silhouette_score")
ax.scatter([4], [s4], s=350, marker="*", c="red", edgecolors="black",
           linewidths=1.2, zorder=5, label=f"本课取 k=4：{s4:.3f}")
ax.axhline(0, color="#888888", linewidth=0.8)
ax.set_title("独立研究③：轮廓系数随 k 变化（bank_customers，四特征标准化）", fontsize=15)
ax.set_xlabel("簇数 k", fontsize=12)
ax.set_ylabel("轮廓系数（-1 到 1）", fontsize=12)
ax.set_xticks(ks)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/16_silhouette.png", dpi=150)
print("\n已保存 outputs/16_silhouette.png")
