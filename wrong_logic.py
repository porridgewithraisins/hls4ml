import numpy as np

# Reproduce the incorrect Conv2DTranspose behavior observed in the HLS build.
# The layer3 tensor mimics the output after an unintended extra transpose,
# so running the scatter below yields the mismatched 19/29/14/25 result.
layer3 = np.array(
    [
        [[1.0, 2.0], [1.0, 2.0]],
        [[2.0, 1.0], [3.0, 2.0]],
    ]
)

weights = np.zeros((3, 3, 1, 2))
weights[..., 0, 0] = np.array(
    [
        [1.0, 2.0, 3.0],
        [0.0, 1.0, 4.0],
        [2.0, 1.0, 1.0],
    ]
)
weights[..., 0, 1] = np.array(
    [
        [1.0, 0.0, 1.0],
        [2.0, 2.0, 3.0],
        [0.0, 1.0, 1.0],
    ]
)
bias = np.array([1.0])

out = np.zeros((2, 2, 1)) + bias
for ih in range(2):
    for iw in range(2):
        for cc in range(2):
            in_val = layer3[ih, iw, cc]
            for fh in range(3):
                for fw in range(3):
                    oh = ih + fh - 1
                    ow = iw + fw - 1
                    if 0 <= oh < 2 and 0 <= ow < 2:
                        out[oh, ow, 0] += in_val * weights[fh, fw, 0, cc]

print(out[..., 0])
print(out.reshape(-1))
