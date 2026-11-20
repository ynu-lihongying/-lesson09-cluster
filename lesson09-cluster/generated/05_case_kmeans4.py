# -*- coding: utf-8 -*-
"""
实践指南九 · 案例 Step1：聚类跑通（tag O9-step1）
对 bank_customers.csv 选取 balance / avg_monthly_spend / product_count / overdue_count
四列标准化后做 KMeans(k=4)，簇编号写回 df 新列 cluster，输出各簇人数（检查点1）。

特征选择理由（红线2）：这四列都是行为特征——余额、月均消费刻画资产与活跃度，
持有产品数刻画粘性，逾期次数刻画风险行为；分群结果即"客户价值画像"。
age/gender/city 是人口属性，混入会把"价值分群"变成"年龄分组/地域分组"。
红线1：customer_id 是编号不是特征，绝不进模型。
红线3：random_state=42、n_init=10 显式写；距离类模型必须先标准化。
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
print(f"数据形状：{df.shape[0]} 行 × {df.shape[1]} 列")

# ---------- 四个行为特征（红线1：不含 customer_id） ----------
FEATS = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]
X_raw = df[FEATS]

# ---------- 标准化：每列拉成均值0、方差1，四特征等权参与距离计算 ----------
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ---------- k-均值聚类（k=4，红线3：显式写 random_state=42 与 n_init=10） ----------
km = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = km.fit_predict(X)
df["cluster"] = labels  # 簇编号写回数据框

# ---------- 检查点1：输出各簇人数 ----------
counts = df["cluster"].value_counts().sort_index()
print("\n检查点1 · 各簇人数：")
print(counts.to_string())
print(f"合计：{counts.sum()} 人")

# ---------- 聚类结果可视化：balance × avg_monthly_spend 二维投影着色 ----------
fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
for c in range(4):
    sub = df[df["cluster"] == c]
    ax.scatter(sub["balance"], sub["avg_monthly_spend"],
               s=45, alpha=0.75, color=colors[c], edgecolors="white", linewidths=0.5,
               label=f"簇 {c}（{len(sub)} 人）")

# 质心在标准化空间里，画回原始坐标必须逆变换（红线：度量衡换了，客户没换）
centers_raw = scaler.inverse_transform(km.cluster_centers_)
ax.scatter(centers_raw[:, 0], centers_raw[:, 1],
           marker="*", s=600, c="red", edgecolors="black", linewidths=1.2,
           zorder=5, label="簇中心（逆变换回原始坐标）")

# 每个簇中心旁标注人数
for c in range(4):
    cx, cy = centers_raw[c, 0], centers_raw[c, 1]
    ax.annotate(f"簇{c}：{int((df['cluster'] == c).sum())}人",
                xy=(cx, cy), xytext=(cx, cy + 1500),
                fontsize=11, fontweight="bold", ha="center")

ax.set_title("200 位银行客户 k-均值分群（k=4，四特征标准化后）\n二维投影：账户余额 × 月均消费", fontsize=15)
ax.set_xlabel("账户余额 balance（元）", fontsize=12)
ax.set_ylabel("月均消费 avg_monthly_spend（元）", fontsize=12)
ax.legend(fontsize=10, loc="best")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/07_case_kmeans4.png", dpi=150)
print("\n已保存 outputs/07_case_kmeans4.png")
