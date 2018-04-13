import numpy as np
import matplotlib.pyplot as plt
import math
# #データを生成
# n = 50
# x_data = np.random.rand(n).astype(np.float32)
# y_data = 0.14* x_data**4  -  0.35* x_data**3  +  0.2 * x_data*2 + 0.01

# #ノイズを加える
# y_data = y_data +0.009 * np.random.randn(n)

EPISODES = 100000  # number of episodes
EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.05  # e-greedy threshold end value
EPS_DECAY = 100  # e-greedy threshold decay

decay =  (EPS_START - EPS_END) / EPS_DECAY
a = np.arange(100)
arr = np.array([])

for t in range(100):
	arr = np.append(arr, np.array([EPS_END + (EPS_START - EPS_END) * math.exp(-1. * a[t] / EPS_DECAY)]))
	# np.append(b, np.array([EPS_START - a[t]*decay]))
	# print(b)

plt.scatter(a,arr)
plt.show()

print(a.shape)
print(arr.shape)

# a = np.arange(12)
# np.append(a, [6, 4, 2])

# # print(a)

# arr = np.array([])
# arr = np.append(arr, np.array([0.1, 2, 3]))
# arr = np.append(arr, np.array([4, 5]))
# print(arr)