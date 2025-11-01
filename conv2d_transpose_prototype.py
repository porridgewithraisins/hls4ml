"""
conv2d_transpose_prototype.py

Simple, readable NumPy prototype for Conv2DTranspose (no stride >1, support 'same' and 'valid').
- Input shape: (N, H, W, C_in)
- Weight shape: (KH, KW, C_in, C_out)  # Keras-compatible
- Bias shape: (C_out,) or None

This is intentionally simple and explicit for use during design and debugging.
"""

import numpy as np


def conv2d_transpose(x, w, bias=None, padding="same"):
    """Compute conv2d transpose (deconvolution) with stride=1.

    Args:
        x: numpy array shape (N, H, W, C_in)
        w: numpy array shape (KH, KW, C_in, C_out)
        bias: numpy array shape (C_out,) or None
        padding: 'same' or 'valid'

    Returns:
        out: numpy array shape (N, H_out, W_out, C_out)
    """
    if x.ndim != 4:
        raise ValueError("Input x must be 4D (N, H, W, C_in)")
    if w.ndim != 4:
        raise ValueError("Weights w must be 4D (KH, KW, C_in, C_out)")

    N, H, W, C_in = x.shape
    KH, KW, C_in_w, C_out = w.shape
    assert C_in_w == C_in, "Weight in-channel mismatch"

    stride_h = 1
    stride_w = 1

    if padding == "same":
        pad_h = (KH - 1) // 2
        pad_w = (KW - 1) // 2
    elif padding == "valid":
        pad_h = 0
        pad_w = 0
    else:
        raise ValueError("padding must be 'same' or 'valid'")

    H_out = (H - 1) * stride_h - 2 * pad_h + KH
    W_out = (W - 1) * stride_w - 2 * pad_w + KW

    out = np.zeros((N, H_out, W_out, C_out), dtype=np.result_type(x, w))

    # Direct implementation: each input pixel contributes a KHxKW patch in the output
    for n in range(N):
        for h in range(H):
            for wi in range(W):
                for c_in in range(C_in):
                    val = x[n, h, wi, c_in]
                    if val == 0:
                        continue
                    # add kernel * val into the right location
                    for kh in range(KH):
                        out_h = h * stride_h - pad_h + kh
                        if out_h < 0 or out_h >= H_out:
                            continue
                        for kw in range(KW):
                            out_w = wi * stride_w - pad_w + kw
                            if out_w < 0 or out_w >= W_out:
                                continue
                            # w[kh, kw, c_in, :] has shape (C_out,)
                            out[n, out_h, out_w, :] += val * w[kh, kw, c_in, :]

    if bias is not None:
        out += bias.reshape((1, 1, 1, C_out))

    return out


# Small helper tests
def _test_1d_toy():
    # 1D toy example adapted to 2D with H=1 for clarity
    # Input = [a, b] -> shape (1, 1, 2, 1)
    a = 2.0
    b = 3.0
    x = np.array([[[[a], [b]]]])  # N=1, H=1, W=2, C=1

    # Kernel = [k1, k2, k3] -> represent as KH=1, KW=3
    k1, k2, k3 = 0.5, 1.0, -0.25
    w = np.zeros((1, 3, 1, 1))
    w[0, 0, 0, 0] = k1
    w[0, 1, 0, 0] = k2
    w[0, 2, 0, 0] = k3

    out = conv2d_transpose(x, w, padding="valid")
    # Expected 1D output: [a*k1, a*k2 + b*k1, a*k3 + b*k2, b*k3]
    expected = np.array([[[[a * k1], [a * k2 + b * k1], [a * k3 + b * k2], [b * k3]]]])

    print("\n1D toy test (embedded in 2D):")
    print("out shape:", out.shape)
    print("out:", out)
    print("expected:", expected)
    assert out.shape == expected.shape, "Shape mismatch for 1D toy"
    if not np.allclose(out, expected):
        print("Mismatch! diff=\n", out - expected)
        raise AssertionError("1D toy values mismatch")
    print("1D toy test passed")


def _test_random_small():
    np.random.seed(0)
    N = 2
    H = 4
    W = 3
    C_in = 2
    C_out = 3
    KH = 3
    KW = 3

    x = np.random.randn(N, H, W, C_in).astype(float)
    w = np.random.randn(KH, KW, C_in, C_out).astype(float)
    b = np.random.randn(C_out).astype(float)

    out_same = conv2d_transpose(x, w, bias=b, padding="same")
    out_valid = conv2d_transpose(x, w, bias=b, padding="valid")

    print("\nRandom small test:")
    print("in shape:", x.shape)
    print("w shape:", w.shape)
    print("out_same shape:", out_same.shape)
    print("out_valid shape:", out_valid.shape)

    # Basic sanity checks on shapes
    KH, KW, _, _ = w.shape
    H_out_valid = (H - 1) * 1 - 2 * 0 + KH
    W_out_valid = (W - 1) * 1 - 2 * 0 + KW
    assert out_valid.shape == (N, H_out_valid, W_out_valid, C_out)

    pad_h = (KH - 1) // 2
    pad_w = (KW - 1) // 2
    H_out_same = (H - 1) * 1 - 2 * pad_h + KH
    W_out_same = (W - 1) * 1 - 2 * pad_w + KW
    assert out_same.shape == (N, H_out_same, W_out_same, C_out)

    # Spot-check linearity: scale input by 2 -> output scales by 2
    out_double = conv2d_transpose(2 * x, w, bias=None, padding="same")
    assert np.allclose(
        out_double, 2 * conv2d_transpose(x, w, bias=None, padding="same")
    )

    print("Random small tests passed (basic sanity checks)")


if __name__ == "__main__":
    _test_1d_toy()
    _test_random_small()
    print("\nAll tests passed")
