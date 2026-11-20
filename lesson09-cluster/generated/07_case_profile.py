# -*- coding: utf-8 -*-
"""
实践指南九 · 案例 Step3：分群画像（tag O9-step3，本课高潮）
groupby('cluster') 输出每簇四特征均值表（画像表）＋分组柱状图＋雷达图，
并给每簇起中文业务命名（检查点3：画像表+图+四个中文业务命名）。
红线1：分群结论必须落到业务命名与建议——"第0簇均值高"不是结论，
"高余额风险客群，建议先控风险再谈增值"才是。
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------- 中文显示设置（Windows 可用字体） ----------
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------- 读取数据、标准化、聚类（与 Step1 完全一致，保证可复现） ----------
df = pd.read_csv("data/bank_customers.csv")
FEATS = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]
scaler = StandardScaler()
X = scaler.fit_transform(df[FEATS])
df["cluster"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X)

# ---------- 检查点3·画像表：每簇四特征均值（终端打印） ----------
profile = df.groupby("cluster").agg(
    人数=("customer_id", "count"),
    平均余额元=("balance", "mean"),
    平均月消费元=("avg_monthly_spend", "mean"),
    平均产品数=("product_count", "mean"),
    平均逾期次数=("overdue_count", "mean"),
    平均年龄=("age", "mean"),
    平均开户月数=("account_age_months", "mean"),
).round(2)
print("检查点3 · 各簇画像表（原始单位均值）：")
print(profile.to_string())

# 标准化后的画像（四特征同一把尺，画图用）
prof_std = pd.DataFrame(X, columns=FEATS)
prof_std["cluster"] = df["cluster"]
std_mean = prof_std.groupby("cluster")[FEATS].mean()

# ---------- 四个中文业务命名（基于上方画像人工判定） ----------
BIZ_NAMES = {
    0: "高消费活跃客群",   # 月均消费 2.4 万元全场最高，余额与其他簇接近
    1: "低消费储蓄客群",   # 月均消费仅约 0.7 万元，余额并不低——钱在账上不动
    2: "高余额风险客群",   # 余额最高近 30 万，但平均逾期 3 次全场最高
    3: "多产品忠诚客群",   # 平均持有 6.7 个产品全场最多，逾期 0.1 次最低
}
BIZ_ACTION = {
    0: "信用卡分期、消费贷、商户联名权益，承接其高消费习惯",
    1: "低门槛理财与货币基金、消费唤醒活动，把沉睡余额盘活",
    2: "贷前审查与额度管控先行，风险可控后再做私行潜力转化",
    3: "交叉销售与权益升级、VIP 客户服务，巩固粘性做大 AUM",
}

# ---------- 图1：分组柱状图（标准化均值，四特征可比） ----------
fig, ax = plt.subplots(figsize=(10, 7))
x = np.arange(len(FEATS))
width = 0.2
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
for i, c in enumerate(sorted(std_mean.index)):
    vals = std_mean.loc[c, FEATS].values
    n = int((df["cluster"] == c).sum())
    ax.bar(x + (i - 1.5) * width, vals, width,
           color=colors[i], label=f"簇{c} {BIZ_NAMES[c]}（{n}人）", edgecolor="white")
ax.axhline(0, color="#888888", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(["账户余额", "月均消费", "持有产品数", "逾期次数"], fontsize=12)
ax.set_ylabel("标准化后均值（0=全体平均）", fontsize=12)
ax.set_title("四客群画像对比：分组柱状图（k=4，四特征标准化后均值）", fontsize=15)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("outputs/09_case_profile_bar.png", dpi=150)
print("\n已保存 outputs/09_case_profile_bar.png")

# ---------- 图2：雷达图（每簇一条闭合线，四轴标准化均值） ----------
labels_cn = ["账户余额", "月均消费", "持有产品数", "逾期次数"]
angles = np.linspace(0, 2 * np.pi, len(labels_cn), endpoint=False).tolist()
angles += angles[:1]  # 闭合

fig, ax = plt.subplots(figsize=(10, 7), subplot_kw=dict(polar=True))
for i, c in enumerate(sorted(std_mean.index)):
    vals = std_mean.loc[c, FEATS].tolist()
    vals += vals[:1]
    n = int((df["cluster"] == c).sum())
    ax.plot(angles, vals, linewidth=2, color=colors[i],
            label=f"簇{c} {BIZ_NAMES[c]}（{n}人）")
    ax.fill(angles, vals, color=colors[i], alpha=0.12)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels_cn, fontsize=12)
ax.set_title("四客群画像雷达图（标准化均值，越外圈越高）", fontsize=15, pad=25)
ax.legend(fontsize=10, loc="upper right", bbox_to_anchor=(1.32, 1.1))
plt.tight_layout()
plt.savefig("outputs/10_case_profile_radar.png", dpi=150)
print("已保存 outputs/10_case_profile_radar.png")

# ---------- 生成中文分群报告（业务命名＋建议，红线1） ----------
lines = [
    "# 银行客户分群画像报告（k-均值，k=4，四特征标准化）",
    "",
    "数据：data/bank_customers.csv（200 位客户）；特征：余额、月均消费、持有产品数、逾期次数"
    "（均为行为特征，分群即客户价值画像；customer_id 不进模型）。",
    "",
    "## 各簇画像表（原始单位均值）",
    "",
    profile.to_string(),
    "",
    "## 四客群业务命名与建议",
    "",
]
for c in sorted(BIZ_NAMES):
    row = profile.loc[c]
    pct = row["人数"] / 200 * 100
    lines.append(
        f"- **簇 {c} · {BIZ_NAMES[c]}（{int(row['人数'])} 人，{pct:.0f}%）**："
        f"平均余额 {row['平均余额元']:,.0f} 元、月均消费 {row['平均月消费元']:,.0f} 元、"
        f"持有 {row['平均产品数']:.1f} 个产品、逾期 {row['平均逾期次数']:.1f} 次。"
        f"业务动作：{BIZ_ACTION[c]}。"
    )
lines += [
    "",
    "## 一句话结论",
    "",
    "四簇余额水平接近，真正区分客群的是消费、产品持有与逾期行为——"
    "**高消费活跃客群**做消费金融承接，**低消费储蓄客群**做理财唤醒，"
    "**高余额风险客群**先控风险再谈增值，**多产品忠诚客群**做权益升级巩固粘性。",
]
with open("generated/bank_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("已保存 generated/bank_report.md")
