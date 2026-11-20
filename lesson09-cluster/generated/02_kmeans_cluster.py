# -*- coding: utf-8 -*-
"""
第 2-4 步：k-均值聚类 + 聚类结果可视化 + 分群报告
特征只用 income 和 balance 两列，不做标准化，random_state=42
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans

# ---------- 中文显示设置（Windows 可用字体） ----------
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------- 读取数据 ----------
df = pd.read_csv("data/mall_customers_sim.csv")

# ---------- 第 2 步：k-均值聚类 ----------
# 特征只用 income 和 balance，customer_id 不进模型
X = df[["income", "balance"]]
km = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = km.fit_predict(X)
df["cluster"] = labels  # 簇编号写回数据框

print("各簇人数：")
print(df["cluster"].value_counts().sort_index())

# ---------- 第 3 步：聚类结果可视化 ----------
fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#DD8452", "#4C72B0", "#55A868"]

# 按簇分别画散点，每个点按所属簇着不同颜色
for c in range(3):
    sub = df[df["cluster"] == c]
    ax.scatter(sub["income"], sub["balance"],
               s=50, alpha=0.7, color=colors[c], edgecolors="white", linewidths=0.5,
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
    ax.annotate(f"簇{c}：{n}人", xy=(cx, cy), xytext=(cx + 1, cy + 2),
                fontsize=11, fontweight="bold", color="black")

ax.set_title("300 位零售银行客户 k-均值聚类结果（k=3）", fontsize=15)
ax.set_xlabel("年收入 income（万元）", fontsize=12)
ax.set_ylabel("账户平均余额 balance（万元）", fontsize=12)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/03_clustered_scatter.png", dpi=150)
print("\n已保存 outputs/03_clustered_scatter.png")

# ---------- 第 4 步：分群报告 ----------
summary = df.groupby("cluster").agg(
    人数=("customer_id", "count"),
    平均年龄=("age", "mean"),
    平均收入万=("income", "mean"),
    平均余额万=("balance", "mean"),
    平均持有产品数=("product_count", "mean"),
).round(2)

# 按平均收入排序，给每簇起业务名字
order = summary["平均收入万"].sort_values().index.tolist()
name_pool = ["低收低余客群", "中收中余客群", "高收高余客群"]
biz_names = {cl: nm for cl, nm in zip(order, name_pool)}

# 生成中文报告
lines = [
    "# 客户分群分析报告（k-均值，k=3）",
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
        f"平均年收入 {row['平均收入万']} 万元、平均余额 {row['平均余额万']} 万元、"
        f"平均持有 {row['平均持有产品数']} 个产品。"
    )

lines += [
    "",
    "## 一句话结论",
    "",
    f"300 位客户在「收入-余额」平面上自然分成三块："
    f"**{biz_names[order[0]]}**（{int(summary.loc[order[0]]['人数'])} 人，"
    f"适合开户与首卡渗透的潜力客户）、"
    f"**{biz_names[order[1]]}**（{int(summary.loc[order[1]]['人数'])} 人，"
    f"适合交叉销售理财与基金的银行基本盘）、"
    f"**{biz_names[order[2]]}**（{int(summary.loc[order[2]]['人数'])} 人，"
    f"适合专属客户经理与私行服务的高价值客户）。",
]

with open("generated/mall_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("已保存 generated/mall_report.md")
