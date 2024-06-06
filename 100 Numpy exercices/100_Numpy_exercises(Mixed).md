# 100 numpy exercises

This is a collection of exercises that have been collected in the numpy mailing list, on stack overflow
and in the numpy documentation. The goal of this collection is to offer a quick reference for both old
and new users but also to provide a set of exercises for those who teach.

If you find an error or think you've a better way to solve some of them, feel
free to open an issue at <https://github.com/rougier/numpy-100>.
File automatically generated. See the documentation to update questions/answers/hints programmatically.

# 100个NumPy练习

这是一系列从NumPy邮件列表、Stack Overflow以及NumPy文档中收集来的练习题。本系列练习的目的是为新老用户提供一个快速参考，同时也为教学人员提供一套练习题。

如果你发现错误或者认为你有更好的解题方式，请随时在<https://github.com/rougier/numpy-100>上提出。文件自动生成。请参阅文档以编程方式更新问题/答案/提示。

#### 1. Import the numpy package under the name `np`（导入名为`np`的numpy包） (★☆☆)

#### 2. Print the numpy version and the configuration（打印numpy的版本和配置） (★☆☆)

#### 3. Create a null vector of size 10（创建一个大小为10的空向量） (★☆☆)

#### 4. How to find the memory size of any array（如何找到任意数组的内存大小？） (★☆☆)

#### 5. How to get the documentation of the numpy add function from the command line?（如何从命令行获取numpy加法函数的文档？） (★☆☆)

#### 6. Create a null vector of size 10 but the fifth value which is 1（创建一个大小为10的空向量，但第五个值为1） (★☆☆)

#### 7. Create a vector with values ranging from 10 to 49（创建一个值从10到49的向量） (★☆☆)

#### 8. Reverse a vector (first element becomes last)（反转一个向量（第一个元素变成最后一个）） (★☆☆)

#### 9. Create a 3x3 matrix with values ranging from 0 to 8（创建一个值从0到8的3x3矩阵） (★☆☆)

#### 10. Find indices of non-zero elements from [1,2,0,0,4,0]（从[1,2,0,0,4,0]中找到非零元素的索引） (★☆☆)

#### 11. Create a 3x3 identity matrix（创建一个3x3的单位矩阵） (★☆☆)

#### 12. Create a 3x3x3 array with random values（创建一个3x3x3的随机值数组） (★☆☆)

#### 13. Create a 10x10 array with random values and find the minimum and maximum values（创建一个10x10的随机值数组，并找到最小值和最大值） (★☆☆)

#### 14. Create a random vector of size 30 and find the mean value（创建一个大小为30的随机向量，并找到其平均值） (★☆☆)

#### 15. Create a 2d array with 1 on the border and 0 inside（创建一个2D数组，边界为1，内部为0） (★☆☆)

#### 16. How to add a border (filled with 0's) around an existing array?（如何给现有数组添加一个边界（用0填充）？） (★☆☆)

#### 17. What is the result of the following expression?（以下表达式的结果是什么？） (★☆☆)
```python
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
np.nan in set([np.nan])
0.3 == 3 * 0.1
```

#### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal（创建一个5x5的矩阵，其值1,2,3,4仅在对角线以下） (★☆☆)

#### 19. Create a 8x8 matrix and fill it with a checkerboard pattern（创建一个8x8的矩阵，并用棋盘格模式填充） (★☆☆)

#### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?（考虑一个形状为(6,7,8)的数组，第100个元素的索引(x,y,z)是什么？） (★☆☆)

#### 21. Create a checkerboard 8x8 matrix using the tile function（使用`tile`函数创建一个8x8的棋盘矩阵） (★☆☆)

#### 22. Normalize a 5x5 random matrix（规范化一个5x5的随机矩阵） (★☆☆)

#### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA)（创建一个自定义的`dtype`，用于描述颜色，由四个无符号字节（RGBA）组成） (★☆☆)

#### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product)（将一个5x3的矩阵乘以一个3x2的矩阵（实际矩阵乘积）） (★☆☆)

#### 25. Given a 1D array, negate all elements which are between 3 and 8, in place.（给定一个1D数组，原地（in place）否定（即取反）所有在3到8之间的元素） (★☆☆)

#### 26. What is the output of the following script?（以下脚本的输出是什么？） (★☆☆)
```python
# Author: Jake VanderPlas

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```

#### 27. Consider an integer vector Z, which of these expressions are legal?（考虑一个整数向量Z，以下哪些表达式是合法的？） (★☆☆)
```python
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
```

#### 28. What are the result of the following expressions?（以下表达式的结果是什么？） (★☆☆)
```python
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)
```

#### 29. How to round away from zero a float array ?（如何对浮点数数组进行向零取整？） (★☆☆)

#### 30. How to find common values between two arrays?（如何找到两个数组之间的共同值？） (★☆☆)

#### 31. How to ignore all numpy warnings (not recommended)?（如何忽略所有的NumPy警告（不推荐）？） (★☆☆)

#### 32. Is the following expressions true?（以下表达式是否成立？） (★☆☆)
```python
np.sqrt(-1) == np.emath.sqrt(-1)
```

#### 33. How to get the dates of yesterday, today and tomorrow?（如何获取昨天、今天和明天的日期？） (★☆☆)

#### 34. How to get all the dates corresponding to the month of July 2016?（如何获取2016年7月的所有日期？） (★★☆)

#### 35. How to compute `((A+B)*(-A/2))` in place (without copy)?（如何原地（不复制）计算表达式 `((A+B)*(-A/2))`？） (★★☆)

#### 36. Extract the integer part of a random array of positive numbers using 4 different methods（使用四种不同的方法提取一个随机正数数组的整数部分） (★★☆)

#### 37. Create a 5x5 matrix with row values ranging from 0 to 4（创建一个5x5的矩阵，行值范围从0到4） (★★☆)

#### 38. Consider a generator function that generates 10 integers and use it to build an array（考虑一个生成10个整数的生成器函数，并使用它构建一个数组） (★☆☆)

#### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded（创建一个大小为10的向量，值从0到1（不包括0和1）） (★★☆)

#### 40. Create a random vector of size 10 and sort it（创建一个大小为10的随机向量并对其进行排序） (★★☆)

#### 41. How to sum a small array faster than np.sum?（如何比使用 `np.sum` 更快地求一个小数组的和？） (★★☆)

#### 42. Consider two random array A and B, check if they are equal（考虑两个随机数组 A 和 B，检查它们是否相等） (★★☆)

#### 43. Make an array immutable (read-only)（使数组不可变（只读）） (★★☆)

#### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates（考虑一个表示笛卡尔坐标的随机 10x2 矩阵，将其转换为极坐标） (★★☆)

#### 45. Create random vector of size 10 and replace the maximum value by 0（创建一个大小为 10 的随机向量，并将最大值替换为 0） (★★☆)

#### 46. Create a structured array with `x` and `y` coordinates covering the [0,1]x[0,1] area（创建一个结构化数组，包含覆盖 [0,1]x[0,1] 区域的 `x` 和 `y` 坐标） (★★☆)

#### 47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj))（给定两个数组 X 和 Y，构造柯西矩阵 C（Cij = 1/(xi - yj)）） (★★☆)

#### 48. Print the minimum and maximum representable value for each numpy scalar type（打印每个 NumPy 浮点标量类型可表示的最小值和最大值） (★★☆)

#### 49. How to print all the values of an array?（如何打印数组的所有值？） (★★☆)

#### 50. How to find the closest value (to a given scalar) in a vector?（如何在向量中找到最接近给定标量的值？） (★★☆)

#### 51. Create a structured array representing a position (x,y) and a color (r,g,b)（创建一个表示位置（x,y）和颜色（r,g,b）的结构化数组） (★★☆)

#### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances（考虑一个形状为 (100,2) 的随机向量，表示坐标，找出逐点距离） (★★☆)

#### 53. How to convert a float (32 bits) array into an integer (32 bits) in place?（如何原地将一个浮点数（32位）数组转换为整数（32位）？）

#### 54. How to read the following file?（如何读取以下文件？） (★★☆)
```
1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11
```

#### 55. What is the equivalent of enumerate for numpy arrays?（NumPy 数组中 `enumerate` 的等价物是什么？） (★★☆)

#### 56. Generate a generic 2D Gaussian-like array（生成一个通用的 2D 类似高斯数组） (★★☆)

#### 57. How to randomly place p elements in a 2D array?（如何在 2D 数组中随机放置 p 个元素？） (★★☆)

#### 58. Subtract the mean of each row of a matrix（减去矩阵每行的平均值） (★★☆)

#### 59. How to sort an array by the nth column?（如何按第 n 列对数组进行排序？） (★★☆)

#### 60. How to tell if a given 2D array has null columns?（如何判断给定的 2D 数组是否有空列？） (★★☆)

#### 61. Find the nearest value from a given value in an array（在数组中查找给定值的最接近值） (★★☆)

#### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator?（考虑两个形状分别为 (1,3) 和 (3,1) 的数组，如何使用迭代器计算它们的和？） (★★☆)

#### 63. Create an array class that has a name attribute（创建一个具有名称属性的数组类） (★★☆)

#### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)?（考虑一个给定的向量，如何根据第二个向量索引的每个元素加 1（注意重复的索引）？） (★★★)

#### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)?（如何根据索引列表 (I) 将向量 (X) 的元素累加到数组 (F) 中？） (★★★)

#### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors（考虑一个形状为 (w,h,3) 的图像（dtype=ubyte），计算其中唯一颜色的数量） (★★☆)

#### 67. Considering a four dimensions array, how to get sum over the last two axis at once?（考虑一个四维数组，如何一次性获取最后两个轴的和？） (★★★)

#### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset  indices?（考虑一个一维向量 D，如何使用一个相同大小的向量 S（描述子集索引）来计算 D 的子集均值？） (★★★)

#### 69. How to get the diagonal of a dot product?（如何获取点积的对角线？） (★★★)

#### 70. Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value?（考虑向量 [1, 2, 3, 4, 5]，如何构建一个在每个值之间插入 3 个连续零的新向量？） (★★★)

#### 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)?（考虑一个维度为 (5,5,3) 的数组，如何将其与一个维度为 (5,5) 的数组相乘？） (★★★)

#### 72. How to swap two rows of an array?（如何交换数组中的两行？） (★★★)

#### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the  triangles（考虑一组描述 10 个三角形（具有共享顶点）的 10 个三元组，如何找到组成所有三角形的唯一线段集合？） (★★★)

#### 74. Given a sorted array C that corresponds to a bincount, how to produce an array A such that np.bincount(A) == C?（给定一个对应于 bincount 的已排序数组 C，如何生成一个数组 A，使得 np.bincount(A) 等于 C？） (★★★)

#### 75. How to compute averages using a sliding window over an array?（如何使用滑动窗口计算数组的平均值？） (★★★)

#### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z[0],Z[1],Z[2]) and each subsequent row is  shifted by 1 (last row should be (Z[-3],Z[-2],Z[-1])（考虑一个一维数组 Z，构建一个二维数组，其第一行为 (Z[0],Z[1],Z[2])，并且每一后续行都向右移动一个位置（最后一行应为 (Z[-3],Z[-2],Z[-1])）？） (★★★)

#### 77. How to negate a boolean, or to change the sign of a float inplace?（如何就地取反一个布尔值或改变浮点数的符号？） (★★★)

#### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i (P0[i],P1[i])?（考虑两组点 P0,P1 描述二维直线和一个点 p，如何计算点 p 到每条直线 i（P0[i],P1[i]）的距离？） (★★★)

#### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P[j]) to each line i (P0[i],P1[i])?（考虑两组点 P0,P1 描述二维直线和一组点 P，如何计算每个点 j（P[j]）到每条直线 i（P0[i],P1[i]）的距离？） (★★★)

#### 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary)（考虑一个任意数组，编写一个函数来提取一个固定形状的子部分，并以给定的元素为中心（必要时用 `fill` 值填充）？） (★★★)

#### 81. Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]?（考虑一个数组 Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]，如何生成数组 R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]？） (★★★)

#### 82. Compute a matrix rank（计算矩阵的秩） (★★★)

#### 83. How to find the most frequent value in an array?（如何找到数组中出现最频繁的值？）

#### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix（从一个随机的 10x10 矩阵中提取所有连续的 3x3 块） (★★★)

#### 85. Create a 2D array subclass such that Z[i,j] == Z[j,i]（创建一个二维数组的子类，使得 Z[i,j] 等于 Z[j,i]） (★★★)

#### 86. Consider a set of p matrices with shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1))（考虑一组形状为 (n,n) 的 p 个矩阵和一组形状为 (n,1) 的 p 个向量。如何一次性计算 p 个矩阵乘积的和？（结果形状为 (n,1)）） (★★★)

#### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)?（考虑一个 16x16 的数组，如何获取块和（块的大小为 4x4）？） (★★★)

#### 88. How to implement the Game of Life using numpy arrays?（如何使用 numpy 数组实现“生命游戏”？） (★★★)

#### 89. How to get the n largest values of an array（如何获取数组中的 n 个最大值？） (★★★)

#### 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item)（给定任意数量的向量，构建笛卡尔积（每个元素的每个组合）） (★★★)

#### 91. How to create a record array from a regular array?（如何从常规数组创建一个记录数组？） (★★★)

#### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods（考虑一个大型向量 Z，使用三种不同的方法计算 Z 的三次方） (★★★)

#### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B?（考虑两个形状分别为 (8,3) 和 (2,2) 的数组 A 和 B。如何找到 A 中包含 B 中每一行元素的行，不考虑 B 中元素的顺序？） (★★★)

#### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3])（考虑一个 10x3 的矩阵，提取具有不等值（例如 [2,2,3]）的行） (★★★)

#### 95. Convert a vector of ints into a matrix binary representation（将一个整数向量转换为二进制表示的矩阵） (★★★)

#### 96. Given a two dimensional array, how to extract unique rows?（给定一个二维数组，如何提取唯一的行？） (★★★)

#### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function（考虑两个向量 A 和 B，编写 einsum 等效的内部、外部、求和以及乘法函数） (★★★)

#### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples（考虑由两个向量（X,Y）描述的路径，如何使用等距样本对其进行采样？） (★★★)?

#### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n.（给定一个整数 n 和一个 2D 数组 X，从 X 中选择可以解释为来自具有 n 个自由度的多项式分布的行，即仅包含整数且和为 n 的行。） (★★★)

#### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means).（计算一维数组 X 的均值的引导 95% 置信区间（即，用替换方式重新采样数组元素 N 次，计算每个样本的均值，然后计算均值的百分位数）。） (★★★)