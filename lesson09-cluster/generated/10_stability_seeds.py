# -*- coding: utf-8 -*-
"""
实践指南九 · 自主实践②：簇的稳定性（15分）
random_state 从 42 换成 0、7、2024 三个值，各跑 k=4 聚类，
对比各簇人数与画像是否大变，写结论。
注意：红线3 要求常规聚类显式 random_state=42；本实验是稳定性实验，
换种子属于实验设计本身，结论仍以 42 为基准对照。
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------- 中文显示设置（Windows 可用字体） ----------
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------- 读取数据并标准化 ----------
df = pd.read_csv("data/bank_customers.csv")
FEATS = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]
scaler = StandardScaler()
X = scaler.fit_transform(df[FEATS])

SEEDS = [42, 0, 7, 2024]
results = {}
print("四个 random_state 下各簇人数与画像（标准化均值）：\n")
for s in SEEDS:
    km = KMeans(n_clusters=4, random_state=s, n_init=10)
    lab = km.fit_predict(X)
    tmp = df.copy()
    tmp["cluster"] = lab
    prof = tmp.groupby("cluster").agg(
        人数=("customer_id", "count"),
        余额=("balance", "mean"),
        月消费=("avg_monthly_spend", "mean"),
        产品数=("product_count", "mean"),
        逾期=("overdue_count", "mean"),
    ).round(2)
    results[s] = (lab, prof)
    print(f"—— random_state={s} ——")
    print(prof.to_string())
    print()

# ---------- 稳定性量化：不同种子分法的两两一致性（调整兰德指数） ----------
from sklearn.metrics import adjusted_rand_score
print("两两 ARI（1=完全一致，越接近 1 越稳定）：")
pairs = []
for i, s1 in enumerate(SEEDS):
    for s2 in SEEDS[i + 1:]:
        ari = adjusted_rand_score(results[s1][0], results[s2][0])
        pairs.append((s1, s2, ari))
        print(f"  seed {s1} vs {s2}: ARI={ari:.3f}")

# ---------- 可视化：三个备选种子的二维投影 ----------
fig, axes = plt.subplots(1, 3, figsize=(10, 7), sharey=True, sharex=True)
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
for ax, s in zip(axes, [0, 7, 2024]):
    lab, prof = results[s]
    for c in range(4):
        sub = df[lab == c]
        ax.scatter(sub["balance"], sub["avg_monthly_spend"], s=22, alpha=0.7,
                   color=colors[c], edgecolors="white", linewidths=0.3,
                   label=f"簇{c}（{len(sub)}人）")
    ax.set_title(f"random_state={s}", fontsize=12)
    ax.set_xlabel("账户余额（元）", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel("月均消费（元）", fontsize=10)
fig.suptitle("自主实践②：换随机种子，分群结构是否大变？（二维投影：余额×月消费）", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("outputs/13_stability.png", dpi=150)
print("\n已保存 outputs/13_stability.png")

# ---------- 结论 ----------
min_ari = min(p[2] for p in pairs)
if min_ari > 0.9:
    verdict = "四个种子分法高度一致（两两 ARI 均 > 0.9），分群结构稳定，画像不大变。"
elif min_ari > 0.5:
    verdict = "分群大体一致但局部有漂移（最小 ARI={:.2f}），画像总体稳定、个别簇边界客户会换簇。".format(min_ari)
else:
    verdict = "分群随种子明显变化（最小 ARI={:.2f}），结构不稳定，需要固定种子并复核特征。".format(min_ari)
print("\n结论：" + verdict)
