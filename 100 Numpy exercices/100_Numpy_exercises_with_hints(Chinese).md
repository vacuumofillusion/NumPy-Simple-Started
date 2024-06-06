# 100个NumPy练习

这是一系列从NumPy邮件列表、Stack Overflow以及NumPy文档中收集来的练习题。本系列练习的目的是为新老用户提供一个快速参考，同时也为教学人员提供一套练习题。

如果你发现错误或者认为你有更好的解题方式，请随时在<https://github.com/rougier/numpy-100>上提出。文件自动生成。请参阅文档以编程方式更新问题/答案/提示。

#### 1. 导入名为`np`的numpy包（★☆☆）

`hint: import … as`

#### 2. 打印numpy的版本和配置（★☆☆）

`hint: np.__version__, np.show_config)`

#### 3. 创建一个大小为10的空向量（★☆☆）

`hint: np.zeros`

#### 4. 如何找到任意数组的内存大小？（★☆☆）

`hint: size, itemsize`

#### 5. 如何从命令行获取numpy加法函数的文档？（★☆☆）

`hint: np.info`

#### 6. 创建一个大小为10的空向量，但第五个值为1（★☆☆）

`hint: array[4]`

#### 7. 创建一个值从10到49的向量（★☆☆）

`hint: arange`

#### 8. 反转一个向量（第一个元素变成最后一个）（★☆☆）

`hint: array[::-1]`

#### 9. 创建一个值从0到8的3x3矩阵（★☆☆）

`hint: reshape`

#### 10. 从[1,2,0,0,4,0]中找到非零元素的索引（★☆☆）

`hint: np.nonzero`

#### 11. 创建一个3x3的单位矩阵（★☆☆）

`hint: np.eye`

#### 12. 创建一个3x3x3的随机值数组（★☆☆）

`hint: np.random.random`

#### 13. 创建一个10x10的随机值数组，并找到最小值和最大值（★☆☆）

`hint: min, max`

#### 14. 创建一个大小为30的随机向量，并找到其平均值（★☆☆）

`hint: mean`

#### 15. 创建一个2D数组，边界为1，内部为0（★☆☆）

`hint: array[1:-1, 1:-1]`

#### 16. 如何给现有数组添加一个边界（用0填充）？（★☆☆）

`hint: np.pad`

#### 17. 以下表达式的结果是什么？（★☆☆）
```python
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
np.nan in set([np.nan])
0.3 == 3 * 0.1
```

`hint: NaN = not a number, inf = infinity`

#### 18. 创建一个5x5的矩阵，其值1,2,3,4仅在对角线以下（★☆☆）

`hint: np.diag`

#### 19. 创建一个8x8的矩阵，并用棋盘格模式填充（★☆☆）

`hint: array[::2]`

#### 20. 考虑一个形状为(6,7,8)的数组，第100个元素的索引(x,y,z)是什么？（★☆☆）

`hint: np.unravel_index`

#### 21. 使用`tile`函数创建一个8x8的棋盘矩阵（★☆☆）

`hint: np.tile`

#### 22. 规范化一个5x5的随机矩阵（★☆☆）

`hint: (x -mean)/std`

#### 23. 创建一个自定义的`dtype`，用于描述颜色，由四个无符号字节（RGBA）组成（★☆☆）

`hint: np.dtype`

#### 24. 将一个5x3的矩阵乘以一个3x2的矩阵（实际矩阵乘积）（★☆☆）

`hint:`

#### 25. 给定一个1D数组，原地（in place）否定（即取反）所有在3到8之间的元素（★☆☆）

`hint: >, <`

#### 26. 以下脚本的输出是什么？（★☆☆）
```python
# 作者：Jake VanderPlas

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```

`hint: np.sum`

#### 27. 考虑一个整数向量Z，以下哪些表达式是合法的？（★☆☆）
```python
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
```

`No hints provided...`

#### 28. 以下表达式的结果是什么？（★☆☆）
```python
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)
```

`No hints provided...`

#### 29. 如何对浮点数数组进行向零取整？（★☆☆）

`hint: np.uniform, np.copysign, np.ceil, np.abs, np.where`

#### 30. 如何找到两个数组之间的共同值？（★☆☆）

`hint: np.intersect1d`

#### 31. 如何忽略所有的NumPy警告（不推荐）？（★☆☆）

`hint: np.seterr, np.errstate`

#### 32. 以下表达式是否成立？（★☆☆）
```python
np.sqrt(-1) == np.emath.sqrt(-1)
```

`hint: imaginary number`

#### 33. 如何获取昨天、今天和明天的日期？（★☆☆）

`hint: np.datetime64, np.timedelta64`

#### 34. 如何获取2016年7月的所有日期？（★★☆）

`hint: np.arange(dtype=datetime64['D'])`

#### 35. 如何原地（不复制）计算表达式 ((A+B)*(-A/2))？（★★☆）

`hint: np.add(out=), np.negative(out=), np.multiply(out=), np.divide(out=)`

#### 36. 使用四种不同的方法提取一个随机正数数组的整数部分（★★☆）

`hint: %, np.floor, astype, np.trunc`

#### 37. 创建一个5x5的矩阵，行值范围从0到4（★★☆）

`hint: np.arange`

#### 38. 考虑一个生成10个整数的生成器函数，并使用它构建一个数组（★☆☆）

`hint: np.fromiter`

#### 39. 创建一个大小为10的向量，值从0到1（不包括0和1）（★★☆）

`hint: np.linspace`

#### 40. 创建一个大小为10的随机向量并对其进行排序（★★☆）

`hint: sort`

#### 41. 如何比使用 `np.sum` 更快地求一个小数组的和？（★★☆）

`hint: np.add.reduce`

#### 42. 考虑两个随机数组 A 和 B，检查它们是否相等（★★☆）

`hint: np.allclose, np.array_equal`

#### 43. 使数组不可变（只读）（★★☆）

`hint: flags.writeable`

#### 44. 考虑一个表示笛卡尔坐标的随机 10x2 矩阵，将其转换为极坐标（★★☆）

`hint: np.sqrt, np.arctan2`

#### 45. 创建一个大小为 10 的随机向量，并将最大值替换为 0（★★☆）

`hint: argmax`

#### 46. 创建一个结构化数组，包含覆盖 [0,1]x[0,1] 区域的 `x` 和 `y` 坐标（★★☆）

`hint: np.meshgrid`

#### 47. 给定两个数组 X 和 Y，构造柯西矩阵 C（Cij = 1/(xi - yj)）（★★☆）

`hint: np.subtract.outer`

#### 48. 打印每个 NumPy 浮点标量类型可表示的最小值和最大值（★★☆）

`hint: np.iinfo, np.finfo, eps`

#### 49. 如何打印数组的所有值？（★★☆）

`hint: np.set_printoptions`

#### 50. 如何在向量中找到最接近给定标量的值？（★★☆）

`hint: argmin`

#### 51. 创建一个表示位置（x,y）和颜色（r,g,b）的结构化数组（★★☆）

`hint: dtype`

#### 52. 考虑一个形状为 (100,2) 的随机向量，表示坐标，找出逐点距离（★★☆）

`hint: np.atleast_2d, T, np.sqrt`

#### 53. 如何原地将一个浮点数（32位）数组转换为整数（32位）？

`hint: view and [:] =`

#### 54. 如何读取以下文件？（★★☆）
```
1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11
```

`hint: np.genfromtxt`

#### 55. NumPy 数组中 `enumerate` 的等价物是什么？（★★☆）

`hint: np.ndenumerate, np.ndindex`

#### 56. 生成一个通用的 2D 类似高斯数组（★★☆）

`hint: np.meshgrid, np.exp`

#### 57. 如何在 2D 数组中随机放置 p 个元素？（★★☆）

`hint: np.put, np.random.choice`

#### 58. 减去矩阵每行的平均值（★★☆）

`hint: mean(axis=,keepdims=)`

#### 59. 如何按第 n 列对数组进行排序？（★★☆）

`hint: argsort`

#### 60. 如何判断给定的 2D 数组是否有空列？（★★☆）

`hint: any, ~`

#### 61. 在数组中查找给定值的最接近值（★★☆）

`hint: np.abs, argmin, flat`

#### 62. 考虑两个形状分别为 (1,3) 和 (3,1) 的数组，如何使用迭代器计算它们的和？（★★☆）

`hint: np.nditer`

#### 63. 创建一个具有名称属性的数组类（★★☆）

`hint: class method`

#### 64. 考虑一个给定的向量，如何根据第二个向量索引的每个元素加 1（注意重复的索引）？（★★★）

`hint: np.bincount | np.add.at`

#### 65. 如何根据索引列表 (I) 将向量 (X) 的元素累加到数组 (F) 中？（★★★）

`hint: np.bincount`

#### 66. 考虑一个形状为 (w,h,3) 的图像（dtype=ubyte），计算其中唯一颜色的数量（★★☆）

`hint: np.unique`

#### 67. 考虑一个四维数组，如何一次性获取最后两个轴的和？（★★★）

`hint: sum(axis=(-2,-1))`

#### 68. 考虑一个一维向量 D，如何使用一个相同大小的向量 S（描述子集索引）来计算 D 的子集均值？（★★★）

`hint: np.bincount`

#### 69. 如何获取点积的对角线？（★★★）

`hint: np.diag`

#### 70. 考虑向量 [1, 2, 3, 4, 5]，如何构建一个在每个值之间插入 3 个连续零的新向量？（★★★）

`hint: array[::4]`

#### 71. 考虑一个维度为 (5,5,3) 的数组，如何将其与一个维度为 (5,5) 的数组相乘？（★★★）

`hint: array[:, :, None]`

#### 72. 如何交换数组中的两行？（★★★）

`hint: array[[]] = array[[]]`

#### 73. 考虑一组描述 10 个三角形（具有共享顶点）的 10 个三元组，如何找到组成所有三角形的唯一线段集合？（★★★）

`hint: repeat, np.roll, np.sort, view, np.unique`

#### 74. 给定一个对应于 bincount 的已排序数组 C，如何生成一个数组 A，使得 np.bincount(A) 等于 C？（★★★）

`hint: np.repeat`

#### 75. 如何使用滑动窗口计算数组的平均值？（★★★）

`hint: np.cumsum, from numpy.lib.stride_tricks import sliding_window_view (np>=1.20.0)`

#### 76. 考虑一个一维数组 Z，构建一个二维数组，其第一行为 (Z[0],Z[1],Z[2])，并且每一后续行都向右移动一个位置（最后一行应为 (Z[-3],Z[-2],Z[-1])）？（★★★）

`hint: from numpy.lib import stride_tricks, from numpy.lib.stride_tricks import sliding_window_view (np>=1.20.0)`

#### 77. 如何就地取反一个布尔值或改变浮点数的符号？（★★★）

`hint: np.logical_not, np.negative`

#### 78. 考虑两组点 P0,P1 描述二维直线和一个点 p，如何计算点 p 到每条直线 i（P0[i],P1[i]）的距离？（★★★）

`No hints provided...`

#### 79. 考虑两组点 P0,P1 描述二维直线和一组点 P，如何计算每个点 j（P[j]）到每条直线 i（P0[i],P1[i]）的距离？（★★★）

`No hints provided...`

#### 80. 考虑一个任意数组，编写一个函数来提取一个固定形状的子部分，并以给定的元素为中心（必要时用 `fill` 值填充）？（★★★）

`hint: minimum maximum`

#### 81. 考虑一个数组 Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]，如何生成数组 R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]？（★★★）

`hint: stride_tricks.as_strided, from numpy.lib.stride_tricks import sliding_window_view (np>=1.20.0)`

#### 82. 计算矩阵的秩（★★★）

`hint: np.linalg.svd, np.linalg.matrix_rank`

#### 83. 如何找到数组中出现最频繁的值？

`hint: np.bincount, argmax`

#### 84. 从一个随机的 10x10 矩阵中提取所有连续的 3x3 块（★★★）

`hint: stride_tricks.as_strided, from numpy.lib.stride_tricks import sliding_window_view (np>=1.20.0)`

#### 85. 创建一个二维数组的子类，使得 Z[i,j] 等于 Z[j,i]（★★★）

`hint: class method`

#### 86. 考虑一组形状为 (n,n) 的 p 个矩阵和一组形状为 (n,1) 的 p 个向量。如何一次性计算 p 个矩阵乘积的和？（结果形状为 (n,1)）（★★★）

`hint: np.tensordot`

#### 87. 考虑一个 16x16 的数组，如何获取块和（块的大小为 4x4）？（★★★）

`hint: np.add.reduceat, from numpy.lib.stride_tricks import sliding_window_view (np>=1.20.0)`

#### 88. 如何使用 numpy 数组实现“生命游戏”？（★★★）

`No hints provided...`

#### 89. 如何获取数组中的 n 个最大值？（★★★）

`hint: np.argsort | np.argpartition`

#### 90. 给定任意数量的向量，构建笛卡尔积（每个元素的每个组合）（★★★）

`hint: np.indices`

#### 91. 如何从常规数组创建一个记录数组？（★★★）

`hint: np.core.records.fromarrays`

#### 92. 考虑一个大型向量 Z，使用三种不同的方法计算 Z 的三次方（★★★）

`hint: np.power, *, np.einsum`

#### 93. 考虑两个形状分别为 (8,3) 和 (2,2) 的数组 A 和 B。如何找到 A 中包含 B 中每一行元素的行，不考虑 B 中元素的顺序？（★★★）

`hint: np.where`

#### 94. 考虑一个 10x3 的矩阵，提取具有不等值（例如 [2,2,3]）的行（★★★）

`No hints provided...`

#### 95. 将一个整数向量转换为二进制表示的矩阵（★★★）

`hint: np.unpackbits`

#### 96. 给定一个二维数组，如何提取唯一的行？（★★★）

`hint: np.ascontiguousarray | np.unique`

#### 97. 考虑两个向量 A 和 B，编写 einsum 等效的内部、外部、求和以及乘法函数（★★★）

`hint: np.einsum`

#### 98. 考虑由两个向量（X,Y）描述的路径，如何使用等距样本对其进行采样？（★★★）

`hint: np.cumsum, np.interp`

#### 99. 给定一个整数 n 和一个 2D 数组 X，从 X 中选择可以解释为来自具有 n 个自由度的多项式分布的行，即仅包含整数且和为 n 的行。（★★★）

`hint: np.logical_and.reduce, np.mod`

#### 100. 计算一维数组 X 的均值的引导 95% 置信区间（即，用替换方式重新采样数组元素 N 次，计算每个样本的均值，然后计算均值的百分位数）。（★★★）

`hint: np.percentile`