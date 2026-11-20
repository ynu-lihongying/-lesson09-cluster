# -*- coding: utf-8 -*-
"""
独立研究②：k-均值对"环形分布"数据会失败（7分）
用 make_circles 生成两个同心圆环玩具数据，同一份数据分别跑
KMeans(k=2) 与 DBSCAN，对比分群结果，一句话对比两者差异。
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import make_circles
from sklearn.cluster import KMeans, DBSCAN

# ---------- 中文显示设置（Windows 可用字体） ----------
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------- 生成同心圆环玩具数据 ----------
X, _ = make_circles(n_samples=600, factor=0.5, noise=0.05, random_state=42)

# ---------- 两种聚类对比 ----------
km = KMeans(n_clusters=2, random_state=42, n_init=10)
lab_km = km.fit_predict(X)
db = DBSCAN(eps=0.2, min_samples=5)
lab_db = db.fit_predict(X)

print(f"KMeans 分群结果：{np.unique(lab_km, return_counts=True)}")
print(f"DBSCAN 分群结果：{np.unique(lab_db, return_counts=True)}"
      f"（-1 为噪声点，共 {(lab_db == -1).sum()} 个）")

# ---------- 可视化对比 ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 7))
colors = ["#4C72B0", "#DD8452", "#55A868"]

# 左图：KMeans——按到质心的距离切，只能切出左右两半"月亮"
for c in np.unique(lab_km):
    axes[0].scatter(X[lab_km == c, 0], X[lab_km == c, 1], s=20, alpha=0.75,
                    color=colors[c], edgecolors="white", linewidths=0.3,
                    label=f"簇{c}")
axes[0].scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                marker="*", s=500, c="red", edgecolors="black", zorder=5, label="质心")
axes[0].set_title("KMeans(k=2)：按距离分堆\n内外环被拦腰切成两半，翻车", fontsize=12)
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

# 右图：DBSCAN——按密度连接，内外两环各自成簇
for c in np.unique(lab_db):
    if c == -1:
        axes[1].scatter(X[lab_db == c, 0], X[lab_db == c, 1], s=20,
                        color="#BBBBBB", alpha=0.5, label="噪声点")
    else:
        axes[1].scatter(X[lab_db == c, 0], X[lab_db == c, 1], s=20, alpha=0.75,
                        color=colors[c % 3], edgecolors="white", linewidths=0.3,
                        label=f"簇{c}")
axes[1].set_title("DBSCAN(eps=0.2)：按密度连接\n内外两环完整分开，正确", fontsize=12)
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

fig.suptitle("独立研究②：环形分布数据——KMeans 翻车 vs DBSCAN 正确（600 点，random_state=42）", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("outputs/15_ring_dbscan.png", dpi=150)
print("\n已保存 outputs/15_ring_dbscan.png")

# ---------- 一句话对比 ----------
print("""
一句话对比：k-均值按"到质心的欧氏距离"分堆，只能切出凸形的月牙块，
识别不了同心环这种非凸结构；DBSCAN 按"密度可达"连接样本，
环上点密集相连各自成簇，所以能完整分开内外两环——
数据长什么形状，决定该用哪把刀。""")
