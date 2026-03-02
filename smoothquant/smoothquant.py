"""
SmoothQuant Demo - 纯 Python 实现

SmoothQuant 是一种训练无关的后训练量化方法，通过数学等价变换将激活的量化难度
迁移到权重上，从而实现 W8A8（8-bit 权重 + 8-bit 激活）量化。

核心公式：
    Y = X·W = (X·diag(s)^(-1)) · (diag(s)·W) = X̂·Ŵ
    平滑因子: s_j = max(|X_j|)^α / max(|W_j|)^(1-α)
    其中 α 控制迁移强度，通常取 0.5
"""

import numpy as np

# 尝试使用 PyTorch，若无则纯 NumPy
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def quantize_symmetric(x, n_bits=8):
    """对称量化：将浮点张量量化为 INT8 范围再反量化"""
    q_max = 2 ** (n_bits - 1) - 1
    scale = np.abs(x).max()
    scale = max(scale, 1e-8)
    scale = scale / q_max
    x_q = np.round(x / scale).astype(np.float32)
    x_deq = x_q * scale
    return x_deq, scale


def quantize_per_channel_weight(w, n_bits=8):
    """按输出通道量化权重 (out_features, in_features)"""
    q_max = 2 ** (n_bits - 1) - 1
    scales = np.abs(w).max(axis=1, keepdims=True)
    scales = np.maximum(scales, 1e-8) / q_max
    w_q = np.round(w / scales).astype(np.float32)
    return w_q * scales, scales


def quantize_per_channel_activation(x, n_bits=8):
    """按输入通道量化激活 (batch, in_features)"""
    q_max = 2 ** (n_bits - 1) - 1
    scales = np.abs(x).max(axis=0, keepdims=True)
    scales = np.maximum(scales, 1e-8) / q_max
    x_q = np.round(x / scales).astype(np.float32)
    return x_q * scales, scales


def get_act_scales_simple(activations):
    """获取激活的每通道最大绝对值 (模拟 calibration)"""
    return np.abs(activations).max(axis=0)


def get_weight_scales_simple(weight):
    """获取权重的每通道（输入维度）最大绝对值"""
    return np.abs(weight).max(axis=0)


def compute_smoothing_factor(act_scales, weight_scales, alpha=0.5):
    """
    计算 SmoothQuant 平滑因子
    s_j = max(|X_j|)^α / max(|W_j|)^(1-α)
    """
    act_scales = np.maximum(act_scales, 1e-8)
    weight_scales = np.maximum(weight_scales, 1e-8)
    s = (act_scales ** alpha) / (weight_scales ** (1 - alpha))
    return np.maximum(s, 1e-8)


def apply_smoothing(x, w, s):
    """
    应用平滑变换：X̂ = X/s, Ŵ = W*s
    保持 Y = X·W = X̂·Ŵ 数学等价
    """
    x_smooth = x / s  # 广播：每列除以对应 s_j
    w_smooth = w * s  # 广播：每行乘以对应 s_j
    return x_smooth, w_smooth


def linear_forward(x, w, b=None):
    """线性层前向: Y = X·W + b"""
    y = x @ w.T
    if b is not None:
        y = y + b
    return y


def demo_smoothquant():
    """SmoothQuant 完整演示"""
    print("=" * 60)
    print("SmoothQuant Demo - 激活平滑 + W8A8 量化")
    print("=" * 60)

    np.random.seed(42)
    batch_size = 32
    in_features = 128
    out_features = 256

    # 1. 构造带 outlier 的激活（模拟 LLM 激活分布）
    print("\n1. 构造测试数据（模拟 LLM 激活中的 outlier）")
    x = np.random.randn(batch_size, in_features).astype(np.float32) * 0.1
    # 在少数通道注入大值 outlier
    outlier_channels = [3, 17, 89]
    for c in outlier_channels:
        x[:, c] *= 50  # 某些通道幅度远大于其他
    print(f"   激活形状: {x.shape}")
    print(f"   激活每通道 max: min={np.abs(x).max(axis=0).min():.2e}, max={np.abs(x).max(axis=0).max():.2e}")

    # 2. 权重（通常较均匀）
    w = np.random.randn(out_features, in_features).astype(np.float32) * 0.02
    print(f"   权重形状: {w.shape}")
    print(f"   权重每通道 max: min={np.abs(w).max(axis=0).min():.2e}, max={np.abs(w).max(axis=0).max():.2e}")

    # 3. FP32 参考输出
    y_fp32 = linear_forward(x, w)
    print(f"\n2. FP32 参考输出: shape={y_fp32.shape}")

    # 4. Naive W8A8 量化（无 SmoothQuant）
    print("\n3. Naive W8A8 量化（无 SmoothQuant）")
    x_naive_q, _ = quantize_per_channel_activation(x)
    w_naive_q, _ = quantize_per_channel_weight(w)
    y_naive = linear_forward(x_naive_q, w_naive_q)
    err_naive = np.abs(y_naive - y_fp32)
    print(f"   输出误差: max={err_naive.max():.2e}, mean={err_naive.mean():.2e}")

    # 5. SmoothQuant：计算平滑因子
    print("\n4. SmoothQuant 平滑")
    act_scales = get_act_scales_simple(x)
    weight_scales = get_weight_scales_simple(w)
    alpha = 0.5
    s = compute_smoothing_factor(act_scales, weight_scales, alpha)
    print(f"   α = {alpha}")
    print(f"   平滑因子 s: min={s.min():.2e}, max={s.max():.2e}")

    # 6. 应用平滑
    x_smooth, w_smooth = apply_smoothing(x, w, s)
    print(f"   平滑后激活每通道 max: min={np.abs(x_smooth).max(axis=0).min():.2e}, max={np.abs(x_smooth).max(axis=0).max():.2e}")

    # 7. SmoothQuant W8A8 量化
    print("\n5. SmoothQuant W8A8 量化")
    x_sq_q, _ = quantize_per_channel_activation(x_smooth)
    w_sq_q, _ = quantize_per_channel_weight(w_smooth)
    y_smoothquant = linear_forward(x_sq_q, w_sq_q)
    err_sq = np.abs(y_smoothquant - y_fp32)
    print(f"   输出误差: max={err_sq.max():.2e}, mean={err_sq.mean():.2e}")

    # 8. 对比
    print("\n6. 误差对比")
    print(f"   Naive W8A8     : max_err={err_naive.max():.2e}, mean_err={err_naive.mean():.2e}")
    print(f"   SmoothQuant W8A8: max_err={err_sq.max():.2e}, mean_err={err_sq.mean():.2e}")
    improvement = (err_naive.mean() - err_sq.mean()) / err_naive.mean() * 100
    print(f"   SmoothQuant 平均误差降低: {improvement:.1f}%")

    print("\n" + "=" * 60)
    print("总结:")
    print("  - 激活中的 outlier 导致 naive 量化误差大")
    print("  - SmoothQuant 通过 s = act^α / weight^(1-α) 将难度迁移到权重")
    print("  - 平滑后激活和权重都更易量化，误差显著降低")
    print("=" * 60)


def demo_alpha_sweep():
    """展示不同 α 对量化误差的影响"""
    print("\n" + "=" * 60)
    print("α 参数扫描（迁移强度）")
    print("=" * 60)

    np.random.seed(42)
    batch_size = 64
    in_features = 64
    out_features = 64
    x = np.random.randn(batch_size, in_features).astype(np.float32) * 0.1
    for c in [5, 20, 40]:
        x[:, c] *= 30
    w = np.random.randn(out_features, in_features).astype(np.float32) * 0.02
    y_fp32 = linear_forward(x, w)

    act_scales = get_act_scales_simple(x)
    weight_scales = get_weight_scales_simple(w)

    print(f"{'α':>6} {'max_err':>12} {'mean_err':>12}")
    print("-" * 32)
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        s = compute_smoothing_factor(act_scales, weight_scales, alpha)
        x_s, w_s = apply_smoothing(x, w, s)
        x_q, _ = quantize_per_channel_activation(x_s)
        w_q, _ = quantize_per_channel_weight(w_s)
        y_q = linear_forward(x_q, w_q)
        err = np.abs(y_q - y_fp32)
        print(f"{alpha:>6.2f} {err.max():>12.2e} {err.mean():>12.2e}")
    print("  α=0: 全部难度在激活; α=1: 全部难度在权重; α=0.5: 均衡")


if __name__ == "__main__":
    demo_smoothquant()
    demo_alpha_sweep()

