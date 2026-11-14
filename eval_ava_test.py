"""
评估 NIMA 在 AVA 官方测试集上的表现。
输入：data/ava_test_predictions.csv（模型预测）
      data/test_labels.csv（人工评分分布，无表头，11 列）
输出：控制台打印 SRCC / PLCC / EMD
运行：python eval_ava_test.py
"""

import numpy as np
import pandas as pd


def emd_distance(p, q):
    """
    单张图片的 EMD（Earth Mover's Distance）。
    与 model/model.py 的 single_emd_loss 数学一致，r=2：
        ( (1/10) * Σ_i |CDF_p(i) - CDF_q(i)|^2 )^(1/2)
    """
    cdf_diff = np.cumsum(p) - np.cumsum(q)
    return float((np.mean(np.abs(cdf_diff) ** 2)) ** 0.5)


def pearson(x, y):
    """皮尔逊相关系数。"""
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    """
    斯皮尔曼秩相关：先转成排名，再算皮尔逊。
    pandas 的 rank 会把并列值取平均秩，这一点对人工均分很关键——
    因为真实评分大量并列在 5.x 附近，处理不好会虚高。
    """
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return pearson(rx, ry)


def main():
    pred = pd.read_csv("data/ava_test_predictions.csv")
    labels = pd.read_csv("data/test_labels.csv", header=None)

    labels.columns = ["image_num"] + [f"gt_vote_{i}" for i in range(1, 11)]
    labels["image_num"] = labels["image_num"].astype(int)

    # 按图片编号对齐，避免顺序不同导致错配
    df = pred.merge(labels, on="image_num", how="inner")
    print(f"预测行数: {len(pred)}  标注行数: {len(labels)}  成功对齐: {len(df)}")
    if len(df) != len(pred) or len(df) != len(labels):
        raise RuntimeError("两边图片编号没有完全对齐，先排查数据问题再做评估")

    gt_probs = df[[f"gt_vote_{i}" for i in range(1, 11)]].to_numpy(dtype=float)
    pred_probs = df[[f"pred_vote_{i}" for i in range(1, 11)]].to_numpy(dtype=float)

    scores = np.arange(1, 11)
    gt_mean = gt_probs @ scores
    pred_mean = pred_probs @ scores

    # 直接复用推理阶段算好的 pred_score，验证计算口径一致
    assert np.allclose(pred_mean, df["pred_score"].to_numpy(), atol=1e-6), \
        "重算的期望分数与 CSV 中的 pred_score 不一致"

    dist_emd = np.array([emd_distance(p, q) for p, q in zip(pred_probs, gt_probs)])

    print("\n========== 分数分布 ==========")
    print(f"预测均分: mean={pred_mean.mean():.4f}  std={pred_mean.std():.4f}  "
          f"min={pred_mean.min():.4f}  max={pred_mean.max():.4f}")
    print(f"真实均分: mean={gt_mean.mean():.4f}  std={gt_mean.std():.4f}  "
          f"min={gt_mean.min():.4f}  max={gt_mean.max():.4f}")

    print("\n========== 核心指标 ==========")
    print(f"SRCC (排序相关性): {spearman(pred_mean, gt_mean):.4f}")
    print(f"PLCC (线性相关性): {pearson(pred_mean, gt_mean):.4f}")
    print(f"EMD  (分布距离)  : {dist_emd.mean():.4f}")


if __name__ == "__main__":
    main()
