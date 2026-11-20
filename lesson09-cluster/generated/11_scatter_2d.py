# -*- coding: utf-8 -*-
"""
实践指南九 · 自主实践③：scatter 二维投影（10分）
沿用案例聚类结果（k=4，四特征标准化，random_state=42），
任选两特征着色 by cluster——这里一次画四组特征对投影，
看同一分群在不同二维视角下的形态。
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------- 中文显示设置（Windows 可用字体） ----------
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------- 读取数据、标准化、聚类（与案例 Step1 完全一致） ----------
df = pd.read_csv("data/bank_customers.csv")
FEATS = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]
X = StandardScaler().fit_transform(df[FEATS])
df["cluster"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X)

# ---------- 四组特征对投影（2×2 画布） ----------
PAIRS = [
    ("balance", "avg_monthly_spend", "账户余额（元）", "月均消费（元）"),
    ("balance", "product_count", "账户余额（元）", "持有产品数"),
    ("avg_monthly_spend", "overdue_count", "月均消费（元）", "逾期次数"),
    ("product_count", "overdue_count", "持有产品数", "逾期次数"),
]
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, (fx, fy, lx, ly) in zip(axes.ravel(), PAIRS):
    for c in range(4):
        sub = df[df["cluster"] == c]
        ax.scatter(sub[fx], sub[fy], s=22, alpha=0.7, color=colors[c],
                   edgecolors="white", linewidths=0.3,
                   label=f"簇{c}（{len(sub)}人）")
    ax.set_xlabel(lx, fontsize=10)
    ax.set_ylabel(ly, fontsize=10)
    ax.set_title(f"{lx} × {ly}", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle("自主实践③：同一分群（k=4）在四组二维投影下的着色散点图", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("outputs/14_scatter_2d.png", dpi=150)
print("已保存 outputs/14_scatter_2d.png")

# ---------- 读图提示（终端） ----------
print("""
读图提示：
· 余额×消费：簇0（高消费）与簇1（低消费）上下分层清晰——这两簇靠消费区分；
· 余额×产品数：簇3（多产品）在右侧形成一条高产品数带；
· 消费×逾期：簇2（高风险）单独悬在逾期>1 的区域，一眼可辨；
· 产品数×逾期：簇2 与簇3 分居两角——同样多产品，一个低风险一个高风险。""")
