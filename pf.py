import numpy as np
import matplotlib.pyplot as plt

def summations(x, y, z):
    xx, yy = x**2 + 2*y + 2, y*2 + z**2 - 3
    return xx, yy

def loss(x, y, z, ta=2, tb=5):
    a, b = summations(x, y, z)
    y = (a-ta)**2 + (b-tb)**2
    return y

def softmax(arr):
    arr = -arr
    arr = np.array([np.exp(arr[i])/np.exp(arr).sum() for i in range(arr.shape[0])])
    return arr

def sigmoid(arr):
    # arr = -arr
    # print(arr)
    arr = np.array([1/(1 + np.exp(arr[i])) for i in range(arr.shape[0])], dtype=np.float64)
    return arr


samples = 10*(np.random.rand(1000, 3) - 0.5)
losses = []
means = []
# print(np.mean(samples, axis=0))
for i in range(10000):
    weights = np.sqrt(np.array([loss(w[0], w[1], w[2]) for w in samples])**2)
    # print(weights)
    # weights = sigmoid(weights)
    # print(weights)
    weights = 1/weights
    weights /= weights.sum()
    post = np.array([weights[i]*samples[i, :] for i in range(samples.shape[0])])
    # print(post)
    new_mean = np.sum(post, axis=0)
    samples = np.random.multivariate_normal(new_mean, np.diag([1/((i+1)), 1/(((i+1))), 1/(((i+1)))]), size=1000)
    losses.append(loss(new_mean[0], new_mean[1], new_mean[2]))
    means.append(new_mean)
    # print(new_mean)


plt.plot(np.log10(np.array(losses))[100:], '*')

plt.plot(np.array(means)[:, 0], 'b*')
plt.plot(np.array(means)[:, 1], 'g*')
plt.plot(np.array(means)[:, 2], 'r*')