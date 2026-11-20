# -*- coding: utf-8 -*-
"""
实践指南九 · 自主实践①：换特征集重新画像（15分）
在原四特征（balance/avg_monthly_spend/product_count/overdue_count）基础上
加入 account_age_months（开户月数），重新标准化聚类（k=4），对比画像与业务解读变化。
红线1：customer_id 仍不进模型；红线3：random_state=42、n_init=10。
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

# ---------- 读取数据 ----------
df = pd.read_csv("data/bank_customers.csv")

def cluster_once(feats, seed=42):
    """给定特征列表，标准化后聚类，返回标准化空间画像与人数"""
    X = StandardScaler().fit_transform(df[feats])
    lab = KMeans(n_clusters=4, random_state=seed, n_init=10).fit_predict(X)
    tmp = pd.DataFrame(X, columns=feats)
    tmp["cluster"] = lab
    return tmp.groupby("cluster")[feats].mean(), pd.Series(lab).value_counts().sort_index()

FEATS4 = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]
FEATS5 = FEATS4 + ["account_age_months"]

mean4, cnt4 = cluster_once(FEATS4)
mean5, cnt5 = cluster_once(FEATS5)

print("【原四特征】标准化画像（k=4）：")
print(mean4.round(2).to_string())
print("\n人数：" + "、".join(f"簇{i}={cnt4[i]}" for i in cnt4.index))
print("\n【加开户月数·五特征】标准化画像（k=4）：")
print(mean5.round(2).to_string())
print("\n人数：" + "、".join(f"簇{i}={cnt5[i]}" for i in cnt5.index))

# ---------- 对比可视化：两个特征集各自的画像柱状图 ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 7), sharey=True)
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

def bar_profile(ax, mean_df, title):
    feats = list(mean_df.columns)
    x = np.arange(len(feats))
    width = 0.2
    for i, c in enumerate(sorted(mean_df.index)):
        n = cnt_all[c]
        ax.bar(x + (i - 1.5) * width, mean_df.loc[c, feats].values, width,
               color=colors[i], edgecolor="white",
               label=f"簇{c}（{n}人）")
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["余额", "月消费", "产品数", "逾期", "开户月数"][:len(feats)],
                       fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)

cnt_all = cnt4
bar_profile(axes[0], mean4, "原四特征：余额/消费/产品/逾期")
cnt_all = cnt5
bar_profile(axes[1], mean5, "五特征：加入开户月数后")
axes[0].set_ylabel("标准化后均值（0=全体平均）", fontsize=11)
fig.suptitle("自主实践①：换特征集（加入开户月数）前后画像对比（k=4）", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("outputs/12_profile_with_age.png", dpi=150)
print("\n已保存 outputs/12_profile_with_age.png")
