# 100个NumPy练习

这是一系列从NumPy邮件列表、Stack Overflow以及NumPy文档中收集来的练习题。本系列练习的目的是为新老用户提供一个快速参考，同时也为教学人员提供一套练习题。

如果你发现错误或者认为你有更好的解题方式，请随时在<https://github.com/rougier/numpy-100>上提出。文件自动生成。请参阅文档以编程方式更新问题/答案/提示。

#### 1. 导入名为`np`的numpy包（★☆☆）

```python
import numpy as np
```

#### 2. 打印numpy的版本和配置（★☆☆）

```python
print(np.__version__)
np.show_config()
```

#### 3. 创建一个大小为10的空向量（★☆☆）

```python
Z = np.zeros(10)
print(Z)
```

#### 4. 如何找到任意数组的内存大小？（★☆☆）

```python
Z = np.zeros((10,10))
print("%d bytes" % (Z.size * Z.itemsize))
```

#### 5. 如何从命令行获取numpy加法函数的文档？（★☆☆）

```python
%run `python -c "import numpy; numpy.info(numpy.add)"`
```

#### 6. 创建一个大小为10的空向量，但第五个值为1（★☆☆）

```python
Z = np.zeros(10)
Z[4] = 1
print(Z)
```

#### 7. 创建一个值从10到49的向量（★☆☆）

```python
Z = np.arange(10,50)
print(Z)
```

#### 8. 反转一个向量（第一个元素变成最后一个）（★☆☆）

```python
Z = np.arange(50)
Z = Z[::-1]
print(Z)
```

#### 9. 创建一个值从0到8的3x3矩阵（★☆☆）

```python
Z = np.arange(9).reshape(3, 3)
print(Z)
```

#### 10. 从[1,2,0,0,4,0]中找到非零元素的索引（★☆☆）

```python
nz = np.nonzero([1,2,0,0,4,0])
print(nz)
```

#### 11. 创建一个3x3的单位矩阵（★☆☆）

```python
Z = np.eye(3)
print(Z)
```

#### 12. 创建一个3x3x3的随机值数组（★☆☆）

```python
Z = np.random.random((3,3,3))
print(Z)
```

#### 13. 创建一个10x10的随机值数组，并找到最小值和最大值（★☆☆）

```python
Z = np.random.random((10,10))
Zmin, Zmax = Z.min(), Z.max()
print(Zmin, Zmax)
```

#### 14. 创建一个大小为30的随机向量，并找到其平均值（★☆☆）

```python
Z = np.random.random(30)
m = Z.mean()
print(m)
```

#### 15. 创建一个2D数组，边界为1，内部为0（★☆☆）

```python
Z = np.ones((10,10))
Z[1:-1,1:-1] = 0
print(Z)
```

#### 16. 如何给现有数组添加一个边界（用0填充）？（★☆☆）

```python
Z = np.ones((5,5))
Z = np.pad(Z, pad_width=1, mode='constant', constant_values=0)
print(Z)

# Using fancy indexing
Z[:, [0, -1]] = 0
Z[[0, -1], :] = 0
print(Z)
```

#### 17. 以下表达式的结果是什么？（★☆☆）
```python
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
np.nan in set([np.nan])
0.3 == 3 * 0.1
```

```python
print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(np.nan in set([np.nan]))
print(0.3 == 3 * 0.1)
```

#### 18. 创建一个5x5的矩阵，其值1,2,3,4仅在对角线以下（★☆☆）

```python
Z = np.diag(1+np.arange(4),k=-1)
print(Z)
```

#### 19. 创建一个8x8的矩阵，并用棋盘格模式填充（★☆☆）

```python
Z = np.zeros((8,8),dtype=int)
Z[1::2,::2] = 1
Z[::2,1::2] = 1
print(Z)
```

#### 20. 考虑一个形状为(6,7,8)的数组，第100个元素的索引(x,y,z)是什么？（★☆☆）

```python
print(np.unravel_index(99,(6,7,8)))
```

#### 21. 使用`tile`函数创建一个8x8的棋盘矩阵（★☆☆）

```python
Z = np.tile( np.array([[0,1],[1,0]]), (4,4))
print(Z)
```

#### 22. 规范化一个5x5的随机矩阵（★☆☆）

```python
Z = np.random.random((5,5))
Z = (Z - np.mean (Z)) / (np.std (Z))
print(Z)
```

#### 23. 创建一个自定义的`dtype`，用于描述颜色，由四个无符号字节（RGBA）组成（★☆☆）

```python
color = np.dtype([("r", np.ubyte),
                  ("g", np.ubyte),
                  ("b", np.ubyte),
                  ("a", np.ubyte)])
```

#### 24. 将一个5x3的矩阵乘以一个3x2的矩阵（实际矩阵乘积）（★☆☆）

```python
Z = np.dot(np.ones((5,3)), np.ones((3,2)))
print(Z)

# Alternative solution, in Python 3.5 and above
Z = np.ones((5,3)) @ np.ones((3,2))
print(Z)
```

#### 25. 给定一个1D数组，原地（in place）否定（即取反）所有在3到8之间的元素（★☆☆）

```python
# Author: Evgeni Burovski

Z = np.arange(11)
Z[(3 < Z) & (Z < 8)] *= -1
print(Z)
```

#### 26. 以下脚本的输出是什么？（★☆☆）
```python
# 作者：Jake VanderPlas

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```

#### 27. 考虑一个整数向量Z，以下哪些表达式是合法的？（★☆☆）
```python
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
```

#### 28. 以下表达式的结果是什么？（★☆☆）
```python
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)
```

#### 29. 如何对浮点数数组进行向零取整？（★☆☆）

```python
# Author: Charles R Harris

Z = np.random.uniform(-10,+10,10)
print(np.copysign(np.ceil(np.abs(Z)), Z))

# More readable but less efficient
print(np.where(Z>0, np.ceil(Z), np.floor(Z)))
```

#### 30. 如何找到两个数组之间的共同值？（★☆☆）

```python
Z1 = np.random.randint(0,10,10)
Z2 = np.random.randint(0,10,10)
print(np.intersect1d(Z1,Z2))
```

#### 31. 如何忽略所有的NumPy警告（不推荐）？（★☆☆）

```python
# Suicide mode on
defaults = np.seterr(all="ignore")
Z = np.ones(1) / 0

# Back to sanity
_ = np.seterr(**defaults)

# Equivalently with a context manager
with np.errstate(all="ignore"):
    np.arange(3) / 0
```

#### 32. 以下表达式是否成立？（★☆☆）
```python
np.sqrt(-1) == np.emath.sqrt(-1)
```

#### 33. 如何获取昨天、今天和明天的日期？（★☆☆）

```python
yesterday = np.datetime64('today') - np.timedelta64(1)
today     = np.datetime64('today')
tomorrow  = np.datetime64('today') + np.timedelta64(1)
```

#### 34. 如何获取2016年7月的所有日期？（★★☆）

```python
Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(Z)
```

#### 35. 如何原地（不复制）计算表达式 ((A+B)*(-A/2))？（★★☆）

```python
A = np.ones(3)*1
B = np.ones(3)*2
np.add(A,B,out=B)
np.divide(A,2,out=A)
np.negative(A,out=A)
np.multiply(A,B,out=A)
```

#### 36. 使用四种不同的方法提取一个随机正数数组的整数部分（★★☆）

```python
Z = np.random.uniform(0,10,10)

print(Z - Z%1)
print(Z // 1)
print(np.floor(Z))
print(Z.astype(int))
print(np.trunc(Z))
```

#### 37. 创建一个5x5的矩阵，行值范围从0到4（★★☆）

```python
Z = np.zeros((5,5))
Z += np.arange(5)
print(Z)

# without broadcasting
Z = np.tile(np.arange(0, 5), (5,1))
print(Z)
```

#### 38. 考虑一个生成10个整数的生成器函数，并使用它构建一个数组（★☆☆）

```python
def generate():
    for x in range(10):
        yield x
Z = np.fromiter(generate(),dtype=float,count=-1)
print(Z)
```

#### 39. 创建一个大小为10的向量，值从0到1（不包括0和1）（★★☆）

```python
Z = np.linspace(0,1,11,endpoint=False)[1:]
print(Z)
```

#### 40. 创建一个大小为10的随机向量并对其进行排序（★★☆）

```python
Z = np.random.random(10)
Z.sort()
print(Z)
```

#### 41. 如何比使用 `np.sum` 更快地求一个小数组的和？（★★☆）

```python
# Author: Evgeni Burovski

Z = np.arange(10)
np.add.reduce(Z)
```

#### 42. 考虑两个随机数组 A 和 B，检查它们是否相等（★★☆）

```python
A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)

# Assuming identical shape of the arrays and a tolerance for the comparison of values
equal = np.allclose(A,B)
print(equal)

# Checking both the shape and the element values, no tolerance (values have to be exactly equal)
equal = np.array_equal(A,B)
print(equal)
```

#### 43. 使数组不可变（只读）（★★☆）

```python
Z = np.zeros(10)
Z.flags.writeable = False
Z[0] = 1
```

#### 44. 考虑一个表示笛卡尔坐标的随机 10x2 矩阵，将其转换为极坐标（★★☆）

```python
Z = np.random.random((10,2))
X,Y = Z[:,0], Z[:,1]
R = np.sqrt(X**2+Y**2)
T = np.arctan2(Y,X)
print(R)
print(T)
```

#### 45. 创建一个大小为 10 的随机向量，并将最大值替换为 0（★★☆）

```python
Z = np.random.random(10)
Z[Z.argmax()] = 0
print(Z)
```

#### 46. 创建一个结构化数组，包含覆盖 [0,1]x[0,1] 区域的 `x` 和 `y` 坐标（★★☆）

```python
Z = np.zeros((5,5), [('x',float),('y',float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,5),
                             np.linspace(0,1,5))
print(Z)
```

#### 47. 给定两个数组 X 和 Y，构造柯西矩阵 C（Cij = 1/(xi - yj)）（★★☆）

```python
# Author: Evgeni Burovski

X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(np.linalg.det(C))
```

#### 48. 打印每个 NumPy 浮点标量类型可表示的最小值和最大值（★★☆）

```python
for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps)
```

#### 49. 如何打印数组的所有值？（★★☆）

```python
np.set_printoptions(threshold=float("inf"))
Z = np.zeros((40,40))
print(Z)
```

#### 50. 如何在向量中找到最接近给定标量的值？（★★☆）

```python
Z = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(Z-v)).argmin()
print(Z[index])
```

#### 51. 创建一个表示位置（x,y）和颜色（r,g,b）的结构化数组（★★☆）

```python
Z = np.zeros(10, [ ('position', [ ('x', float, 1),
                                  ('y', float, 1)]),
                   ('color',    [ ('r', float, 1),
                                  ('g', float, 1),
                                  ('b', float, 1)])])
print(Z)
```

#### 52. 考虑一个形状为 (100,2) 的随机向量，表示坐标，找出逐点距离（★★☆）

```python
Z = np.random.random((10,2))
X,Y = np.atleast_2d(Z[:,0], Z[:,1])
D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
print(D)

# Much faster with scipy
import scipy
# Thanks Gavin Heverly-Coulson (#issue 1)
import scipy.spatial

Z = np.random.random((10,2))
D = scipy.spatial.distance.cdist(Z,Z)
print(D)
```

#### 53. 如何原地将一个浮点数（32位）数组转换为整数（32位）？

```python
# Thanks Vikas (https://stackoverflow.com/a/10622758/5989906)
# & unutbu (https://stackoverflow.com/a/4396247/5989906)
Z = (np.random.rand(10)*100).astype(np.float32)
Y = Z.view(np.int32)
Y[:] = Z
print(Y)
```

#### 54. 如何读取以下文件？（★★☆）
```
1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11
```

```python
from io import StringIO

# Fake file
s = StringIO('''1, 2, 3, 4, 5

                6,  ,  , 7, 8

                 ,  , 9,10,11
''')
Z = np.genfromtxt(s, delimiter=",", dtype=np.int)
print(Z)
```

#### 55. NumPy 数组中 `enumerate` 的等价物是什么？（★★☆）

```python
Z = np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(Z):
    print(index, value)
for index in np.ndindex(Z.shape):
    print(index, Z[index])
```

#### 56. 生成一个通用的 2D 类似高斯数组（★★☆）

```python
X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
D = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0.0
G = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )
print(G)
```

#### 57. 如何在 2D 数组中随机放置 p 个元素？（★★☆）

```python
# Author: Divakar

n = 10
p = 3
Z = np.zeros((n,n))
np.put(Z, np.random.choice(range(n*n), p, replace=False),1)
print(Z)
```

#### 58. 减去矩阵每行的平均值（★★☆）

```python
# Author: Warren Weckesser

X = np.random.rand(5, 10)

# Recent versions of numpy
Y = X - X.mean(axis=1, keepdims=True)

# Older versions of numpy
Y = X - X.mean(axis=1).reshape(-1, 1)

print(Y)
```

#### 59. 如何按第 n 列对数组进行排序？（★★☆）

```python
# Author: Steve Tjoa

Z = np.random.randint(0,10,(3,3))
print(Z)
print(Z[Z[:,1].argsort()])
```

#### 60. 如何判断给定的 2D 数组是否有空列？（★★☆）

```python
# Author: Warren Weckesser

# null : 0 
Z = np.random.randint(0,3,(3,10))
print((~Z.any(axis=0)).any())

# null : np.nan
Z=np.array([
    [0,1,np.nan],
    [1,2,np.nan],
    [4,5,np.nan]
])
print(np.isnan(Z).all(axis=0))
```

#### 61. 在数组中查找给定值的最接近值（★★☆）

```python
Z = np.random.uniform(0,1,10)
z = 0.5
m = Z.flat[np.abs(Z - z).argmin()]
print(m)
```

#### 62. 考虑两个形状分别为 (1,3) 和 (3,1) 的数组，如何使用迭代器计算它们的和？（★★☆）

```python
A = np.arange(3).reshape(3,1)
B = np.arange(3).reshape(1,3)
it = np.nditer([A,B,None])
for x,y,z in it: z[...] = x + y
print(it.operands[2])
```

#### 63. 创建一个具有名称属性的数组类（★★☆）

```python
class NamedArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.name = getattr(obj, 'name', "no name")

Z = NamedArray(np.arange(10), "range_10")
print (Z.name)
```

#### 64. 考虑一个给定的向量，如何根据第二个向量索引的每个元素加 1（注意重复的索引）？（★★★）

```python
# Author: Brett Olsen

Z = np.ones(10)
I = np.random.randint(0,len(Z),20)
Z += np.bincount(I, minlength=len(Z))
print(Z)

# Another solution
# Author: Bartosz Telenczuk
np.add.at(Z, I, 1)
print(Z)
```

#### 65. 如何根据索引列表 (I) 将向量 (X) 的元素累加到数组 (F) 中？（★★★）

```python
# Author: Alan G Isaac

X = [1,2,3,4,5,6]
I = [1,3,9,3,4,1]
F = np.bincount(I,X)
print(F)
```

#### 66. 考虑一个形状为 (w,h,3) 的图像（dtype=ubyte），计算其中唯一颜色的数量（★★☆）

```python
# Author: Fisher Wang

w, h = 256, 256
I = np.random.randint(0, 4, (h, w, 3)).astype(np.ubyte)
colors = np.unique(I.reshape(-1, 3), axis=0)
n = len(colors)
print(n)

# Faster version
# Author: Mark Setchell
# https://stackoverflow.com/a/59671950/2836621

w, h = 256, 256
I = np.random.randint(0,4,(h,w,3), dtype=np.uint8)

# View each pixel as a single 24-bit integer, rather than three 8-bit bytes
I24 = np.dot(I.astype(np.uint32),[1,256,65536])

# Count unique colours
n = len(np.unique(I24))
print(n)
```

#### 67. 考虑一个四维数组，如何一次性获取最后两个轴的和？（★★★）

```python
A = np.random.randint(0,10,(3,4,3,4))
# solution by passing a tuple of axes (introduced in numpy 1.7.0)
sum = A.sum(axis=(-2,-1))
print(sum)
# solution by flattening the last two dimensions into one
# (useful for functions that don't accept tuples for axis argument)
sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
print(sum)
```

#### 68. 考虑一个一维向量 D，如何使用一个相同大小的向量 S（描述子集索引）来计算 D 的子集均值？（★★★）

```python
# Author: Jaime Fernández del Río

D = np.random.uniform(0,1,100)
S = np.random.randint(0,10,100)
D_sums = np.bincount(S, weights=D)
D_counts = np.bincount(S)
D_means = D_sums / D_counts
print(D_means)

# Pandas solution as a reference due to more intuitive code
import pandas as pd
print(pd.Series(D).groupby(S).mean())
```

#### 69. 如何获取点积的对角线？（★★★）

```python
# Author: Mathieu Blondel

A = np.random.uniform(0,1,(5,5))
B = np.random.uniform(0,1,(5,5))

# Slow version
np.diag(np.dot(A, B))

# Fast version
np.sum(A * B.T, axis=1)

# Faster version
np.einsum("ij,ji->i", A, B)
```

#### 70. 考虑向量 [1, 2, 3, 4, 5]，如何构建一个在每个值之间插入 3 个连续零的新向量？（★★★）

```python
# Author: Warren Weckesser

Z = np.array([1,2,3,4,5])
nz = 3
Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
Z0[::nz+1] = Z
print(Z0)
```

#### 71. 考虑一个维度为 (5,5,3) 的数组，如何将其与一个维度为 (5,5) 的数组相乘？（★★★）

```python
A = np.ones((5,5,3))
B = 2*np.ones((5,5))
print(A * B[:,:,None])
```

#### 72. 如何交换数组中的两行？（★★★）

```python
# Author: Eelco Hoogendoorn

A = np.arange(25).reshape(5,5)
A[[0,1]] = A[[1,0]]
print(A)
```

#### 73. 考虑一组描述 10 个三角形（具有共享顶点）的 10 个三元组，如何找到组成所有三角形的唯一线段集合？（★★★）

```python
# Author: Nicolas P. Rougier

faces = np.random.randint(0,100,(10,3))
F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
F = F.reshape(len(F)*3,2)
F = np.sort(F,axis=1)
G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
G = np.unique(G)
print(G)
```

#### 74. 给定一个对应于 bincount 的已排序数组 C，如何生成一个数组 A，使得 np.bincount(A) 等于 C？（★★★）

```python
# Author: Jaime Fernández del Río

C = np.bincount([1,1,2,3,4,4,6])
A = np.repeat(np.arange(len(C)), C)
print(A)
```

#### 75. 如何使用滑动窗口计算数组的平均值？（★★★）

```python
# Author: Jaime Fernández del Río

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
Z = np.arange(20)
print(moving_average(Z, n=3))

# Author: Jeff Luo (@Jeff1999)
# make sure your NumPy >= 1.20.0

from numpy.lib.stride_tricks import sliding_window_view

Z = np.arange(20)
print(sliding_window_view(Z, window_shape=3).mean(axis=-1))
```

#### 76. 考虑一个一维数组 Z，构建一个二维数组，其第一行为 (Z[0],Z[1],Z[2])，并且每一后续行都向右移动一个位置（最后一行应为 (Z[-3],Z[-2],Z[-1])）？（★★★）

```python
# Author: Joe Kington / Erik Rigtorp
from numpy.lib import stride_tricks

def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.strides[0], a.strides[0])
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
Z = rolling(np.arange(10), 3)
print(Z)

# Author: Jeff Luo (@Jeff1999)

Z = np.arange(10)
print(sliding_window_view(Z, window_shape=3))
```

#### 77. 如何就地取反一个布尔值或改变浮点数的符号？（★★★）

```python
# Author: Nathaniel J. Smith

Z = np.random.randint(0,2,100)
np.logical_not(Z, out=Z)

Z = np.random.uniform(-1.0,1.0,100)
np.negative(Z, out=Z)
```

#### 78. 考虑两组点 P0,P1 描述二维直线和一个点 p，如何计算点 p 到每条直线 i（P0[i],P1[i]）的距离？（★★★）

```python
def distance(P0, P1, p):
    T = P1 - P0
    L = (T**2).sum(axis=1)
    U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L
    U = U.reshape(len(U),1)
    D = P0 + U*T - p
    return np.sqrt((D**2).sum(axis=1))

P0 = np.random.uniform(-10,10,(10,2))
P1 = np.random.uniform(-10,10,(10,2))
p  = np.random.uniform(-10,10,( 1,2))
print(distance(P0, P1, p))
```

#### 79. 考虑两组点 P0,P1 描述二维直线和一组点 P，如何计算每个点 j（P[j]）到每条直线 i（P0[i],P1[i]）的距离？（★★★）

```python
# Author: Italmassov Kuanysh

# based on distance function from previous question
P0 = np.random.uniform(-10, 10, (10,2))
P1 = np.random.uniform(-10,10,(10,2))
p = np.random.uniform(-10, 10, (10,2))
print(np.array([distance(P0,P1,p_i) for p_i in p]))
```

#### 80. 考虑一个任意数组，编写一个函数来提取一个固定形状的子部分，并以给定的元素为中心（必要时用 `fill` 值填充）？（★★★）

```python
# Author: Nicolas Rougier

Z = np.random.randint(0,10,(10,10))
shape = (5,5)
fill  = 0
position = (1,1)

R = np.ones(shape, dtype=Z.dtype)*fill
P  = np.array(list(position)).astype(int)
Rs = np.array(list(R.shape)).astype(int)
Zs = np.array(list(Z.shape)).astype(int)

R_start = np.zeros((len(shape),)).astype(int)
R_stop  = np.array(list(shape)).astype(int)
Z_start = (P-Rs//2)
Z_stop  = (P+Rs//2)+Rs%2

R_start = (R_start - np.minimum(Z_start,0)).tolist()
Z_start = (np.maximum(Z_start,0)).tolist()
R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs,0))).tolist()
Z_stop = (np.minimum(Z_stop,Zs)).tolist()

r = [slice(start,stop) for start,stop in zip(R_start,R_stop)]
z = [slice(start,stop) for start,stop in zip(Z_start,Z_stop)]
R[r] = Z[z]
print(Z)
print(R)
```

#### 81. 考虑一个数组 Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]，如何生成数组 R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]？（★★★）

```python
# Author: Stefan van der Walt

Z = np.arange(1,15,dtype=np.uint32)
R = stride_tricks.as_strided(Z,(11,4),(4,4))
print(R)

# Author: Jeff Luo (@Jeff1999)

Z = np.arange(1, 15, dtype=np.uint32)
print(sliding_window_view(Z, window_shape=4))
```

#### 82. 计算矩阵的秩（★★★）

```python
# Author: Stefan van der Walt

Z = np.random.uniform(0,1,(10,10))
U, S, V = np.linalg.svd(Z) # Singular Value Decomposition
rank = np.sum(S > 1e-10)
print(rank)

# alternative solution:
# Author: Jeff Luo (@Jeff1999)

rank = np.linalg.matrix_rank(Z)
print(rank)
```

#### 83. 如何找到数组中出现最频繁的值？

```python
Z = np.random.randint(0,10,50)
print(np.bincount(Z).argmax())
```

#### 84. 从一个随机的 10x10 矩阵中提取所有连续的 3x3 块（★★★）

```python
# Author: Chris Barker

Z = np.random.randint(0,5,(10,10))
n = 3
i = 1 + (Z.shape[0]-3)
j = 1 + (Z.shape[1]-3)
C = stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
print(C)

# Author: Jeff Luo (@Jeff1999)

Z = np.random.randint(0,5,(10,10))
print(sliding_window_view(Z, window_shape=(3, 3)))
```

#### 85. 创建一个二维数组的子类，使得 Z[i,j] 等于 Z[j,i]（★★★）

```python
# Author: Eric O. Lebigot
# Note: only works for 2d array and value setting using indices

class Symetric(np.ndarray):
    def __setitem__(self, index, value):
        i,j = index
        super(Symetric, self).__setitem__((i,j), value)
        super(Symetric, self).__setitem__((j,i), value)

def symetric(Z):
    return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)

S = symetric(np.random.randint(0,10,(5,5)))
S[2,3] = 42
print(S)
```

#### 86. 考虑一组形状为 (n,n) 的 p 个矩阵和一组形状为 (n,1) 的 p 个向量。如何一次性计算 p 个矩阵乘积的和？（结果形状为 (n,1)）（★★★）

```python
# Author: Stefan van der Walt

p, n = 10, 20
M = np.ones((p,n,n))
V = np.ones((p,n,1))
S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
print(S)

# It works, because:
# M is (p,n,n)
# V is (p,n,1)
# Thus, summing over the paired axes 0 and 0 (of M and V independently),
# and 2 and 1, to remain with a (n,1) vector.
```

#### 87. 考虑一个 16x16 的数组，如何获取块和（块的大小为 4x4）？（★★★）

```python
# Author: Robert Kern

Z = np.ones((16,16))
k = 4
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                       np.arange(0, Z.shape[1], k), axis=1)
print(S)

# alternative solution:
# Author: Sebastian Wallkötter (@FirefoxMetzger)

Z = np.ones((16,16))
k = 4

windows = np.lib.stride_tricks.sliding_window_view(Z, (k, k))
S = windows[::k, ::k, ...].sum(axis=(-2, -1))

# Author: Jeff Luo (@Jeff1999)

Z = np.ones((16, 16))
k = 4
print(sliding_window_view(Z, window_shape=(k, k))[::k, ::k].sum(axis=(-2, -1)))
```

#### 88. 如何使用 numpy 数组实现“生命游戏”？（★★★）

```python
# Author: Nicolas Rougier

def iterate(Z):
    # Count neighbours
    N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
         Z[1:-1,0:-2]                + Z[1:-1,2:] +
         Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])

    # Apply rules
    birth = (N==3) & (Z[1:-1,1:-1]==0)
    survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)
    Z[...] = 0
    Z[1:-1,1:-1][birth | survive] = 1
    return Z

Z = np.random.randint(0,2,(50,50))
for i in range(100): Z = iterate(Z)
print(Z)
```

#### 89. 如何获取数组中的 n 个最大值？（★★★）

```python
Z = np.arange(10000)
np.random.shuffle(Z)
n = 5

# Slow
print (Z[np.argsort(Z)[-n:]])

# Fast
print (Z[np.argpartition(-Z,n)[:n]])
```

#### 90. 给定任意数量的向量，构建笛卡尔积（每个元素的每个组合）（★★★）

```python
# Author: Stefan Van der Walt

def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix

print (cartesian(([1, 2, 3], [4, 5], [6, 7])))
```

#### 91. 如何从常规数组创建一个记录数组？（★★★）

```python
Z = np.array([("Hello", 2.5, 3),
              ("World", 3.6, 2)])
R = np.core.records.fromarrays(Z.T,
                               names='col1, col2, col3',
                               formats = 'S8, f8, i8')
print(R)
```

#### 92. 考虑一个大型向量 Z，使用三种不同的方法计算 Z 的三次方（★★★）

```python
# Author: Ryan G.

x = np.random.rand(int(5e7))

%timeit np.power(x,3)
%timeit x*x*x
%timeit np.einsum('i,i,i->i',x,x,x)
```

#### 93. 考虑两个形状分别为 (8,3) 和 (2,2) 的数组 A 和 B。如何找到 A 中包含 B 中每一行元素的行，不考虑 B 中元素的顺序？（★★★）

```python
# Author: Gabe Schwartz

A = np.random.randint(0,5,(8,3))
B = np.random.randint(0,5,(2,2))

C = (A[..., np.newaxis, np.newaxis] == B)
rows = np.where(C.any((3,1)).all(1))[0]
print(rows)
```

#### 94. 考虑一个 10x3 的矩阵，提取具有不等值（例如 [2,2,3]）的行（★★★）

```python
# Author: Robert Kern

Z = np.random.randint(0,5,(10,3))
print(Z)
# solution for arrays of all dtypes (including string arrays and record arrays)
E = np.all(Z[:,1:] == Z[:,:-1], axis=1)
U = Z[~E]
print(U)
# soluiton for numerical arrays only, will work for any number of columns in Z
U = Z[Z.max(axis=1) != Z.min(axis=1),:]
print(U)
```

#### 95. 将一个整数向量转换为二进制表示的矩阵（★★★）

```python
# Author: Warren Weckesser

I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
B = ((I.reshape(-1,1) & (2**np.arange(8))) != 0).astype(int)
print(B[:,::-1])

# Author: Daniel T. McDonald

I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
print(np.unpackbits(I[:, np.newaxis], axis=1))
```

#### 96. 给定一个二维数组，如何提取唯一的行？（★★★）

```python
# Author: Jaime Fernández del Río

Z = np.random.randint(0,2,(6,3))
T = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
_, idx = np.unique(T, return_index=True)
uZ = Z[idx]
print(uZ)

# Author: Andreas Kouzelis
# NumPy >= 1.13
uZ = np.unique(Z, axis=0)
print(uZ)
```

#### 97. 考虑两个向量 A 和 B，编写 einsum 等效的内部、外部、求和以及乘法函数（★★★）

```python
# Author: Alex Riley
# Make sure to read: http://ajcr.net/Basic-guide-to-einsum/

A = np.random.uniform(0,1,10)
B = np.random.uniform(0,1,10)

np.einsum('i->', A)       # np.sum(A)
np.einsum('i,i->i', A, B) # A * B
np.einsum('i,i', A, B)    # np.inner(A, B)
np.einsum('i,j->ij', A, B)    # np.outer(A, B)
```

#### 98. 考虑由两个向量（X,Y）描述的路径，如何使用等距样本对其进行采样？（★★★）

```python
# Author: Bas Swinckels

phi = np.arange(0, 10*np.pi, 0.1)
a = 1
x = a*phi*np.cos(phi)
y = a*phi*np.sin(phi)

dr = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths
r = np.zeros_like(x)
r[1:] = np.cumsum(dr)                # integrate path
r_int = np.linspace(0, r.max(), 200) # regular spaced path
x_int = np.interp(r_int, r, x)       # integrate path
y_int = np.interp(r_int, r, y)
```

#### 99. 给定一个整数 n 和一个 2D 数组 X，从 X 中选择可以解释为来自具有 n 个自由度的多项式分布的行，即仅包含整数且和为 n 的行。（★★★）

```python
# Author: Evgeni Burovski

X = np.asarray([[1.0, 0.0, 3.0, 8.0],
                [2.0, 0.0, 1.0, 1.0],
                [1.5, 2.5, 1.0, 0.0]])
n = 4
M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1)
M &= (X.sum(axis=-1) == n)
print(X[M])
```

#### 100. 计算一维数组 X 的均值的引导 95% 置信区间（即，用替换方式重新采样数组元素 N 次，计算每个样本的均值，然后计算均值的百分位数）。（★★★）

```python
# Author: Jessica B. Hamrick

X = np.random.randn(100) # random 1D array
N = 1000 # number of bootstrap samples
idx = np.random.randint(0, X.size, (N, X.size))
means = X[idx].mean(axis=1)
confint = np.percentile(means, [2.5, 97.5])
print(confint)
```
