import numpy as np

# input (N=1, C_in=2, H=W=2)
x = np.array([[
    [[1., 2.],
     [2., 1.]],
    [[1., 3.],
     [2., 2.]],
]])

# weights (C_in, C_out=1, kH=kW=3)
w = np.zeros((2, 1, 3, 3))
w[0, 0] = np.array([[1., 2., 3.],
                    [0., 1., 4.],
                    [2., 1., 1.]])
w[1, 0] = np.array([[1., 0., 1.],
                    [2., 2., 3.],
                    [0., 1., 1.]])
bias = np.array([1.])

stride = 1
padding = 1
kH = kW = 3
N, Cin, H, W = x.shape
H_out = (H - 1) * stride - 2 * padding + kH  # -> 2
W_out = (W - 1) * stride - 2 * padding + kW  # -> 2

out = np.zeros((N, 1, H_out, W_out))
for n in range(N):
  for c in range(Cin):
    for i in range(H):
      for j in range(W):
        for ki in range(kH):
          for kj in range(kW):
            oi = i*stride - padding + ki
            oj = j*stride - padding + kj
            if 0 <= oi < H_out and 0 <= oj < W_out:
              out[n, 0, oi, oj] += x[n, c, i, j] * w[c, 0, ki, kj]

out += bias.reshape(1, 1, 1, 1)

print(out[0, 0])          # 2x2
print(out[0, 0].ravel())  # flattened
