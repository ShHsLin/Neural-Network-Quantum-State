import numpy as np

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale*np.where(x>=0.0, x, alpha*np.exp(x)-alpha)


x = np.random.normal(size=(300, 200))
for _ in range(100):
    w = np.random.normal(size=(200, 200), scale=np.sqrt(1/200))  # their initialization scheme
    x = selu(np.dot(x, w))
    m = np.mean(x, axis=1)
    s = np.std(x, axis=1)
    print(m.min(), m.max(), s.min(), s.max())

