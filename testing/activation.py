import numpy as np
import matplotlib.pyplot as plt


def elu(x, alpha=1.):
    return (x>=0)*x + (x<0) * alpha * (np.exp(x)-1)

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale*np.where(x>=0.0, x, alpha*np.exp(x)-alpha)

def softplus2(x):
    return np.log( (1+np.exp(x))/2. )

def softplus(x):
    return np.log( (1+np.exp(x)) )

def relu(x):
    return x * (x > 0)

def prelu(x, a=0.05):
    return np.maximum(x, x*a)

act = relu

x=np.arange(-5,5,0.1)
plt.plot(x, act(x))
# plt.show()

print("test alpha for %s " % act)
x = np.random.normal(size=(int(1e7), 1))
before_m = np.mean(x**2, axis=0)
before_s = np.std(x**2, axis=0)
# x = softplus2(x)
x = act(x)
after_m = np.mean(x**2, axis=0)
after_s = np.std(x**2, axis=0)
alpha =  (after_m/before_m)
print("alpha = %f " % (alpha) )

# Below is a numerical testing of the grow in signal magnitude #
x = np.random.normal(size=(300, 200))
print("mean.min, mean.max, std.min, std.max")
for _ in range(50):
    w = np.random.normal(size=(200, 200), scale=np.sqrt(1/alpha/200))  # their initialization scheme
    x = act(np.dot(x, w))
    # x = selu(np.dot(x, w))
    m = np.mean(x, axis=1)
    s = np.std(x, axis=1)
    print(m.min(), m.max(), s.min(), s.max())


