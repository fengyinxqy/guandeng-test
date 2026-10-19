"""Plot audited CSV outputs; never refit a policy or invent business data."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

BLUE, ORANGE, GREY, PURPLE = "#2563a6", "#c45c35", "#9ba8b5", "#8b75ab"
CLUSTERS = ["华东边缘集群", "华南算力主集群", "香港低延迟专线", "华北异构集群"]
RULES = ["no_action", "flag_all", "age_only", "slots_only", "global_or", "cluster_policy"]
RULE_LABELS = ["不干预", "全部标记", "仅年龄 >15 分钟", "仅槽位 ≤3", "全局 OR", "分集群规则"]


def configure_font():
    for name in ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "WenQuanYi Zen Hei"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams["font.family"] = name
        break
    else:
        raise RuntimeError("A Chinese font is required: Microsoft YaHei or Noto Sans CJK SC")
    plt.rcParams.update({
        "font.size": 11, "axes.unicode_minus": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#cbd5df", "axes.labelcolor": "#324457",
        "xtick.color": "#526172", "ytick.color": "#324457",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "grid.alpha": 0.22,
    })


def save(fig, path):
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"Generated: {path}")


def plot_risks(directory):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8))
    fig.suptitle("集群风险差异：状态过期与低槽位需要分开看", fontsize=18, x=0.06, ha="left", y=0.97)
    specs = [
        ("cluster_age_groups.csv", "stale_gt_15m", ["年龄 ≤15 分钟", "年龄 >15 分钟"], "状态年龄", axes[0]),
        ("cluster_slot_groups.csv", "slots_le_3", ["槽位 >3", "槽位 ≤3"], "剩余槽位", axes[1]),
    ]
    for filename, flag, labels, title, ax in specs:
        data = pd.read_csv(directory / filename)
        assert data.known_outcomes.sum() == 976 and data.failures.sum() == 126
        for i, risk in enumerate([False, True]):
            group = data[data[flag].eq(risk)].set_index("service_cluster").loc[CLUSTERS]
            ys = [j + (i - 0.5) * 0.32 for j in range(len(CLUSTERS))]
            rates = group.failure_rate_known * 100
            ax.barh(ys, rates, height=0.28, label=labels[i], color=[BLUE, ORANGE][i])
            for y, (_, row) in zip(ys, group.iterrows()):
                rate = row.failure_rate_known * 100
                ax.text(rate + 1.0, y, f"{rate:.1f}%  ({int(row.failures)}/{int(row.known_outcomes)})", va="center", fontsize=10)
        ax.set_title(f"按{title}分组", loc="left", fontsize=13, pad=16)
        ax.set_yticks(range(4), CLUSTERS)
        ax.invert_yaxis()
        ax.set_xlim(0, 103)
        ax.set_xticks([0, 20, 40, 60, 80, 100], ["0%", "20%", "40%", "60%", "80%", "100%"])
        ax.set_xlabel("已知结果中的失败率")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.19), ncol=2, frameon=False, fontsize=10)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.82, bottom=0.28, wspace=0.43)
    fig.text(0.06, 0.10, "全量描述：去重后 1027 次请求，976 次结果已知，51 次未知不进入失败率分母。括号 = 失败数 / 已知结果数。", fontsize=10)
    fig.text(0.06, 0.055, "来源：cluster_age_groups.csv、cluster_slot_groups.csv。分组为观察性关联，不能据此证明因果或定位坏节点。", fontsize=10, color="#617080")
    save(fig, directory / "risk_by_cluster.png")


def plot_policy(directory):
    table = pd.read_csv(directory / "policy_comparison.csv")
    data = table[table.split.eq("validation")].set_index("rule").loc[RULES]
    assert data.requests.eq(310).all() and data.failures.eq(36).all()
    assert (data.alerted_failures + data.alerted_successes + data.alerted_unknowns).eq(data.alerts).all()
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 7), sharey=True, gridspec_kw={"width_ratios": [1.25, 1]})
    fig.suptitle("后段回放：同样覆盖 34 次失败，分集群规则少标记 37 次", fontsize=18, x=0.06, ha="left", y=0.97)
    y = list(range(6))
    base = [0] * 6
    for column, label, color in [
        ("alerted_failures", "原失败", ORANGE),
        ("alerted_successes", "原成功", GREY),
        ("alerted_unknowns", "结果未知", PURPLE),
    ]:
        values = data[column].to_numpy()
        left.barh(y, values, left=base, height=0.58, color=color, label=label)
        base = [a + b for a, b in zip(base, values)]
    for i, (_, row) in enumerate(data.iterrows()):
        left.text(row.alerts + 4, i, f"{int(row.alerts)} 次｜覆盖 {int(row.alerted_failures)}/36", va="center", fontsize=10)
    left.set_yticks(y, RULE_LABELS)
    left.invert_yaxis()
    left.set_xlim(0, 450)
    left.set_xlabel("标记请求数（按历史结果分解）")
    left.set_title("干预范围与失败覆盖", loc="left", fontsize=13, pad=16)
    left.legend(loc="lower right", frameon=False, fontsize=10)
    colors = [GREY] * 5 + [BLUE]
    right.barh(y, data.scenario_loss_yuan, height=0.58, color=colors)
    for i, cost in enumerate(data.scenario_loss_yuan):
        right.text(cost + 65, i, f"{cost:,.0f} 元", va="center", color="#324457")
    right.set_xlim(0, 5300)
    right.set_xlabel("情景总成本（元，越低越好）")
    right.set_title("乐观情景成本：q=100%，r=0%", loc="left", fontsize=13, pad=16)
    for ax in [left, right]:
        ax.grid(axis="x")
        ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.82, bottom=0.24, wspace=0.16)
    fig.text(0.06, 0.13, "后段 310 次：36 失败、261 成功、13 未知。原成功被标记不等于节点异常；分集群规则仍会干预 41 次原成功。", fontsize=10)
    fig.text(0.06, 0.085, "成本 = 未覆盖的已知失败 ×120 + 标记请求 ×10；假设命中失败全部可挽回，且干预不引入新失败。", fontsize=10)
    fig.text(0.06, 0.04, "来源：policy_comparison.csv。10 元/标记请求是事件费代理；上述收益非线上实测，未知结果不计额外收益。", fontsize=10, color="#617080")
    save(fig, directory / "rule_comparison.png")


def plot_sensitivity(directory):
    table = pd.read_csv(directory / "sensitivity.csv")
    data = table[table.split.eq("validation")]
    fig, ax = plt.subplots(figsize=(12, 6.8))
    fig.suptitle("改派能挽回多少失败，决定策略是否值得上线", fontsize=18, x=0.08, ha="left", y=0.96)
    for rate, color, style in [(0, BLUE, "-"), (0.01, GREY, "--"), (0.05, ORANGE, "-.")]:
        group = data[data.new_failure_rate.eq(rate)].sort_values("rescue_rate")
        ax.plot(group.rescue_rate * 100, group.scenario_saving_yuan, marker="o", color=color, linestyle=style, linewidth=2, label=f"原成功请求新增失败比例 r={rate:.0%}")
    reference = data[data.rescue_rate.eq(1) & data.new_failure_rate.eq(0)].iloc[0]
    boundary = reference.break_even_rescue_rate * 100
    ax.axhline(0, color="#475569", linewidth=1)
    ax.axvline(boundary, color=BLUE, linestyle=":", linewidth=1.3)
    ax.text(boundary + 1.4, 3320, f"r=0 时回本点\nq={boundary:.2f}%", color=BLUE, va="top", fontsize=11)
    half = data[data.rescue_rate.eq(0.5) & data.new_failure_rate.eq(0)].iloc[0]
    ax.annotate(f"q=50%，r=0\n净收益 {half.scenario_saving_yuan:,.0f} 元", xy=(50, half.scenario_saving_yuan), xytext=(38, 2150), arrowprops={"arrowstyle": "->", "color": BLUE}, color=BLUE)
    ax.text(99, reference.scenario_saving_yuan + 120, f"乐观情景 {reference.scenario_saving_yuan:,.0f} 元", ha="right", color=BLUE)
    ax.set_xlim(0, 103)
    ax.set_ylim(-1200, 3800)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("原会失败请求被挽回的比例 q（假设值）")
    ax.set_ylabel("相对不干预的净收益（元）")
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.83, bottom=0.24)
    fig.text(0.08, 0.13, "后段固定规则：标记 79 次，命中 34 次原失败、41 次原成功、4 次未知。净收益 = 120×(34q−41r)−790。", fontsize=10)
    fig.text(0.08, 0.085, "来源：sensitivity.csv。q、r 均未被现有数据观测；曲线是情景测算，不是实验结果或成功率预测。", fontsize=10)
    fig.text(0.08, 0.04, "动作费按每个标记请求 10 元代理；未知结果不计收益或新增失败，所有标记均计费。", fontsize=10, color="#617080")
    save(fig, directory / "cost_sensitivity.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    configure_font()
    plot_risks(args.output_dir)
    plot_policy(args.output_dir)
    plot_sensitivity(args.output_dir)


if __name__ == "__main__":
    main()
