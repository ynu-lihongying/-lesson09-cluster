# -*- coding: utf-8 -*-
"""
基于元计数据重跑客户分群（balance 单位为元，income 仍为万元）
聚类参数不变：k=3，random_state=42，不做标准化
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans

# ---------- 中文显示设置（Windows 可用字体） ----------
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------- 1. 数据初检 ----------
df = pd.read_csv("data/mall_customers_yuan.csv")
print(f"行列数：{df.shape}")
print("\n各列缺失值：")
print(df.isnull().sum().to_string())
print("\n数值列基本统计：")
print(df[["age", "income", "balance", "product_count"]].describe().round(2).to_string())

# ---------- 2. 重跑聚类（k=3、random_state=42，不标准化） ----------
X = df[["income", "balance"]]  # 特征只用 income 和 balance
km = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = km.fit_predict(X)  # 簇编号写回数据框

print("\n各簇人数与特征均值：")
counts = df["cluster"].value_counts().sort_index()
for c, n in counts.items():
    sub = df[df["cluster"] == c]
    print(f"  簇 {c}：{n} 人 | 平均收入 {sub['income'].mean():.1f} 万 | "
          f"平均余额 {sub['balance'].mean():,.0f} 元")

# ---------- 3. 重画聚类图 ----------
fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#DD8452", "#4C72B0", "#55A868"]

# 按簇分别画散点，按簇着色
for c in range(3):
    sub = df[df["cluster"] == c]
    ax.scatter(sub["income"], sub["balance"],
               s=40, alpha=0.7, color=colors[c], edgecolors="white", linewidths=0.5,
               label=f"簇 {c}（{len(sub)} 人）")

# 簇中心画大星形标记
centers = km.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1],
           marker="*", s=600, c="red", edgecolors="black", linewidths=1.2,
           zorder=5, label="簇中心")

# 在每个簇中心附近标注该簇人数
for c in range(3):
    cx, cy = centers[c, 0], centers[c, 1]
    n = int((df["cluster"] == c).sum())
    ax.annotate(f"簇{c}：{n}人", xy=(cx, cy), xytext=(cx + 1, cy + 3000),
                fontsize=11, fontweight="bold", color="black")

ax.set_title("300 位零售银行客户 k-均值聚类结果（元计数据，未标准化）", fontsize=15)
ax.set_xlabel("年收入 income（万元）", fontsize=12)
ax.set_ylabel("账户平均余额 balance（元）", fontsize=12)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/02_yuan_fail.png", dpi=150)
print("\n已保存 outputs/02_yuan_fail.png")

# ---------- 4. 更新分群报告 ----------
summary = df.groupby("cluster").agg(
    人数=("customer_id", "count"),
    平均年龄=("age", "mean"),
    平均收入万=("income", "mean"),
    平均余额元=("balance", "mean"),
    平均持有产品数=("product_count", "mean"),
).round(2)

# 按平均余额排序，给每簇起业务名字
order = summary["平均余额元"].sort_values().index.tolist()
name_pool = ["低余额客群", "中余额客群", "高余额客群"]
biz_names = {cl: nm for cl, nm in zip(order, name_pool)}

# 生成中文报告
lines = [
    "# 客户分群分析报告（元计数据重跑，k=3，未标准化）",
    "",
    "## 各簇统计概览",
    "",
    summary.to_string(),
    "",
    "## 客群业务命名",
    "",
]
for cl in order:
    row = summary.loc[cl]
    pct = row["人数"] / 300 * 100
    lines.append(
        f"- **簇 {cl} · {biz_names[cl]}**：共 {int(row['人数'])} 人"
        f"（占 {pct:.1f}%），平均年龄 {row['平均年龄']} 岁、"
        f"平均年收入 {row['平均收入万']} 万元、"
        f"平均余额 {row['平均余额元']:,.0f} 元、"
        f"平均持有 {row['平均持有产品数']} 个产品。"
    )

lines += [
    "",
    "## 一句话结论",
    "",
    f"300 位客户在「收入-余额」平面上分成三块："
    f"**{biz_names[order[0]]}**（{int(summary.loc[order[0]]['人数'])} 人）、"
    f"**{biz_names[order[1]]}**（{int(summary.loc[order[1]]['人数'])} 人）、"
    f"**{biz_names[order[2]]}**（{int(summary.loc[order[2]]['人数'])} 人）。",
]

with open("generated/mall_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("已更新 generated/mall_report.md")
