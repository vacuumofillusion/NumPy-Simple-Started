# NumPy: 绝对基础入门指南

欢迎来到NumPy绝对初学者指南！

NumPy（**Num**erical **Py**thon）是一个广泛应用于科学和工程领域的开源Python库。NumPy库包含了多维数组数据结构，如同类的N维`ndarray`，以及一套在此类数据结构上高效操作的大型函数库。欲了解更多关于NumPy的信息，请访问[什么是NumPy](https://numpy.org/devdocs/user/whatisnumpy.html)，如有评论或建议，请[联系我们](https://numpy.org/community/)！

## 如何导入NumPy

安装NumPy后，可以通过以下方式将其导入Python代码中：

```python
import numpy as np
```

这一广泛应用的约定允许通过简短且易于识别的前缀（`np.`）访问NumPy特性，同时区分NumPy特性与同名的其他特性。

## 阅读示例代码

在NumPy文档中，你会看到如下形式的代码块：

```python
>>> a = np.array([[1, 2, 3],
...               [4, 5, 6]])
>>> a.shape
(2, 3)
```

由`>>>`或`...`引导的文本是**输入**，即你在脚本中或Python提示符下输入的代码。其余部分是**输出**，即运行你的代码后得到的结果。请注意，`>>>`和`...`并非代码的一部分，如果在Python提示符下输入可能会引发错误。

## 为何使用NumPy？

Python列表是非常优秀的通用容器。它们可以是“异构”的，意味着可以包含多种类型的元素，并且当对少量元素执行个别操作时速度相当快。

根据数据的特性和需要执行的操作类型，其他容器可能更为合适；通过利用这些特性，我们可以提高速度、减少内存消耗，并为执行各种常见处理任务提供高级语法。当有大量的“同质”（同一类型）数据需要在CPU上处理时，NumPy大放异彩。

## 什么是“数组”？

在计算机编程中，数组是一种用于存储和检索数据的结构。我们常把数组想象成一个空间中的网格，每个单元格子存储数据的一个元素。例如，如果数据的每个元素都是数字，我们可能会将一维数组形象化为一个列表：

$$\begin{aligned}
\begin{array}{|c|c|c|c|}
\hline
1 & 5 & 2 & 0 \\
\hline
\end{array}
\end{aligned}$$

二维数组就像一张表格：

$$\begin{aligned}
\begin{array}{|c|c|c|c|}
\hline
1 & 5 & 2 & 0 \\
\hline
8 & 3 & 6 & 1 \\
\hline
1 & 7 & 2 & 9 \\
\hline
\end{array}
\end{aligned}$$

三维数组则像是一组表格，也许像是印在不同页面上的堆叠起来一样。在NumPy中，这个概念被泛化到任意数量的维度，因此基本的数组类被称为`ndarray`：它代表了一个“N维数组”。

大多数NumPy数组都有一定的限制。例如：

- 数组内的所有元素必须是相同类型的数据。
- 创建后，数组的总大小不能改变。
- 形必须是“矩形”，而不是“锯齿状”；例如，二维数组的每一行必须有相同数量的列。

当满足这些条件时，NumPy利用这些特性使数组比其他限制较少的数据结构更快、内存效率更高且使用更方便。

在本文档剩余部分，我们将使用“数组”一词来指代指`ndarray`的实例。

## 数组基础

初始化数组的一种方法是使用Python序列，如列表。例如：

```python
>>> a = np.array([1, 2, 3, 4, 5, 6])
>>> a
array([1, 2, 3, 4, 5, 6])
```

数组的元素可以通过[多种方式](https://numpy.org/devdocs/user/quickstart.html#quickstart-indexing-slicing-and-iterating)进行访问。例如，我们可以像访问原始列表中的元素一样访问这个数组中的单个元素：使用方括号内的元素的整数索引。

```python
>>> a[0]
1
```

> 与内置Python序列一样，NumPy数组是“0索引”的：数组的第一个元素使用索引`0`访问，而不是`1`。

与原始列表一样，数组是可变的。

```python
>>> a[0] = 10
>>> a
array([10,  2,  3,  4,  5,  6])
```

与原始列表类似，Python的切片表示法也可以用于索引。

```python
>>> a[:3]
array([10, 2, 3])
```

一个主要区别是，列表的切片索引会将元素复制到一个新列表中，但数组的切片返回一个*视图*：一个指向原始数组中数据的对象。原始数组可以通过这个视图进行变更。

```python
>>> b = a[3:]
>>> b
array([4, 5, 6])
>>> b[0] = 40
>>> a
array([ 10,  2,  3, 40,  5,  6])
```

请查看[Copies and views](https://numpy.org/devdocs/user/basics.copies.html#basics-copies-and-views)以获取关于数组操作何时返回视图而非副本的更全面解释。

二维及更高维度的数组可以从嵌套的Python序列初始化：

```python
>>> a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
>>> a
array([[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12]])
```

在NumPy中，数组的维度有时被称为“轴”。这种术语可能有助于区分数组的维度和数组所表示的数据的维度。例如，数组`a`可能表示三个点，每个点都位于一个四维空间中，但`a`只有两个“轴”。

数组与列表的列表之间的另一个区别是，可以通过在单个方括号内指定每个轴的索引（以逗号分隔）来访问数组的元素。例如，元素`8`位于第`1`行和第`3`列：

```python
>>> a[1, 3]
8
```

在数学中，通常先通过行索引再通过列索引来引用矩阵的元素。这对于二维数组也适用，但更好的心理模型是将列索引视为*最后*一个，行索引视为*倒数第二个*。这种思维可以推广到具有*任何*数量的维度的数组。

你可能会听到0维（零维）数组被称为“标量”，1维（一维）数组被称为“向量”，2维（二维）数组被称为“矩阵”，或者N维（N维，其中“N”通常是大于2的整数）数组被称为“张量”。为了清晰起见，在引用数组时最好避免使用这些数学术语，因为这些名称所代表的数学对象的行为与数组不同（例如，“矩阵”乘法与“数组”乘法在本质上是不同的），而且科学Python生态系统中还有其他具有这些名称的对象（例如，PyTorch的基本数据结构就是“张量”）。

## 数组属性

*本节介绍了数组的* `ndim`、`shape`、`size` *和* `dtype` *属性*。

------------------------------------------------------------------------

数组的维度数量包含在 `ndim` 属性中。

```python
>>> a.ndim
2
```

数组的形状是一个非负整数的元组，它指定了每个维度上的元素数量。

```python
>>> a.shape
(3, 4)
>>> len(a.shape) == a.ndim
True
```

数组中固定的、总的元素数量包含在 `size` 属性中。

```python
>>> a.size
12
>>> import math
>>> a.size == math.prod(a.shape)
True
```

数组通常是“同质的”，这意味着它们只包含一种“数据类型”的元素。数据类型记录在 `dtype` 属性中。

```python
>>> a.dtype
dtype('int64')  # "int" 代表整数，"64" 代表 64 位
```

[在这里了解更多关于数组属性的信息](https://numpy.org/devdocs/reference/arrays.ndarray.html#arrays-ndarray) 并学习[关于数组对象的更多内容](https://numpy.org/devdocs/reference/arrays.html#arrays)。

## 如何创建基础数组

*本节介绍了* `np.zeros()`、`np.ones()`、`np.empty()`、`np.arange()`、`np.linspace()`

------------------------------------------------------------------------

除了从元素序列中创建数组外，你还可以轻松地创建一个填充了`0`的数组：

```python
>>> np.zeros(2)
array([0., 0.])
```

或者一个填充了`1`的数组：

```python
>>> np.ones(2)
array([1., 1.])
```

甚至是一个空数组！`empty`函数会创建一个初始内容为随机的数组，这取决于内存的状态。使用`empty`而不是`zeros`（或类似函数）的原因是速度——只需确保之后填充每个元素即可！

```python
>>> # 创建一个包含2个元素的空数组
>>> np.empty(2) #doctest: +SKIP
array([3.14, 42.  ])  # 内容可能不同
```

你可以创建一个包含一系列元素的数组：

```python
>>> np.arange(4)
array([0, 1, 2, 3])
```

你甚至可以创建一个包含一系列等间隔区间的数组。为此，你需要指定**起始数字**、**结束数字**和**步长**。

```python
>>> np.arange(2, 9, 2)
array([2, 4, 6, 8])
```

你还可以使用`np.linspace()`来创建一个在指定区间内线性分布的值的数组：

```python
>>> np.linspace(0, 10, num=5)
array([ 0. ,  2.5,  5. ,  7.5, 10. ])
```

**指定你的数据类型**

虽然默认的数据类型是浮点数（`np.float64`），但你可以使用`dtype`关键字明确指定你想要的数据类型。

```python
>>> x = np.ones(2, dtype=np.int64)
>>> x
array([1, 1])
```

[在这里了解更多关于创建数组的信息](https://numpy.org/devdocs/user/quickstart.html#quickstart-array-creation)

## 添加、删除和排序元素

*本节涵盖了* `np.sort()`、`np.concatenate()`

------------------------------------------------------------------------

使用`np.sort()`可以简单地排序数组。在调用函数时，你可以指定轴、排序方式和排序顺序。

如果你从以下数组开始：

```python
>>> arr = np.array([2, 1, 5, 3, 7, 4, 6, 8])
```

你可以快速地将数字按升序排序：

```python
>>> np.sort(arr)
array([1, 2, 3, 4, 5, 6, 7, 8])
```

除了返回数组的排序副本的`sort`外，你还可以使用：

-   [`argsort`](https://numpy.org/devdocs/reference/generated/numpy.argsort.html#numpy.argsort)，它沿着指定轴进行间接排序，
-   [`lexsort`](https://numpy.org/devdocs/reference/generated/numpy.lexsort.html#numpy.lexsort)，它基于多个键进行稳定的间接排序，
-   [`searchsorted`](https://numpy.org/devdocs/reference/generated/numpy.searchsorted.html#numpy.searchsorted)，它会在已排序的数组中找到元素，
-   [`partition`](https://numpy.org/devdocs/reference/generated/numpy.partition.html#numpy.partition)，它执行部分排序。

要了解更多关于数组排序的信息，请参阅：[`sort`](https://numpy.org/devdocs/reference/generated/numpy.sort.html#numpy.sort)。

如果你从以下数组开始：

```python
>>> a = np.array([1, 2, 3, 4])
>>> b = np.array([5, 6, 7, 8])
```

你可以使用`np.concatenate()`将它们连接起来：

```python
>>> np.concatenate((a, b))
array([1, 2, 3, 4, 5, 6, 7, 8])
```

或者，如果你从以下数组开始：

```python
>>> x = np.array([[1, 2], [3, 4]])
>>> y = np.array([[5, 6]])
```

你可以使用以下方式将它们连接起来：

```python
>>> np.concatenate((x, y), axis=0)
array([[1, 2],
       [3, 4],
       [5, 6]])
```

要从数组中删除元素，简单地使用索引来选择你想要保留的元素即可。

要了解更多关于`concatenate`的信息，请参阅：[`concatenate`](https://numpy.org/devdocs/reference/generated/numpy.concatenate.html#numpy.concatenate)。

## 如何知道数组的形状和大小？

*本节涵盖* `ndarray.ndim`，`ndarray.size`，`ndarray.shape`

------------------------------------------------------------------------

`ndarray.ndim` 将告诉你数组的轴或维度的数量。

`ndarray.size` 将告诉你数组的总元素数量。这是数组形状中元素的乘积。

`ndarray.shape` 将显示一个整数元组，表示沿数组的每个维度存储的元素数量。例如，如果你有一个2-D数组，它有2行和3列，那么你的数组的形状就是 `(2, 3)`。

例如，如果你创建了这个数组：

```python
>>> array_example = np.array([[[0, 1, 2, 3],
...                            [4, 5, 6, 7]],
...
...                           [[0, 1, 2, 3],
...                            [4, 5, 6, 7]],
...
...                           [[0, 1, 2, 3],
...                            [4, 5, 6, 7]]])
```

要查找数组的维度数量，请运行：

```python
>>> array_example.ndim
3
```

要查找数组中的总元素数量，请运行：

```python
>>> array_example.size
24
```

而要查找你的数组的形状，请运行：

```python
>>> array_example.shape
(3, 2, 4)
```

## 可以重塑数组的形状吗？

*本节涵盖* `arr.reshape()`

------------------------------------------------------------------------

**可以！**

使用 `arr.reshape()` 可以为数组赋予新的形状，同时不改变数据。只需要记住，当你使用重塑方法时，你希望产生的数组需要具有与原始数组相同数量的元素。如果你从一个包含12个元素的数组开始，你需要确保你的新数组也总共有12个元素。

如果你从以下数组开始：

```python
>>> a = np.arange(6)
>>> print(a)
[0 1 2 3 4 5]
```

你可以使用 `reshape()` 来重塑你的数组。例如，你可以将这个数组重塑为一个有三行两列的数组：

```python
>>> b = a.reshape(3, 2)
>>> print(b)
[[0 1]
 [2 3]
 [4 5]]
```

使用 `np.reshape`，你可以指定一些可选参数：

```python
>>> np.reshape(a, shape=(1, 6), order='C')
array([[0, 1, 2, 3, 4, 5]])
```

`a` 是要被重塑的数组。

`newshape` 是你想要的新形状。你可以指定一个整数或一个整数元组。如果你指定一个整数，结果将是一个具有该长度的数组。这个形状应该与原始形状兼容。

`order:` `C` 意味着使用类似C的索引顺序来读取/写入元素，`F` 意味着使用类似Fortran的索引顺序来读取/写入元素，`A` 意味着如果`a`在内存中是Fortran连续的，则使用类似Fortran的索引顺序读取/写入元素，否则使用类似C的索引顺序。（这是一个可选参数，不需要指定。）

如果你想了解更多关于C和Fortran顺序的信息，你可以[在这里阅读更多关于NumPy数组内部组织的内容](https://numpy.org/devdocs/dev/internals.html#numpy-internals)。基本上，C和Fortran顺序与索引如何对应数组在内存中的存储顺序有关。在Fortran中，当遍历二维数组的元素时，这些元素按其在内存中的存储顺序移动，**第一个**索引是最快变化的索引。随着第一个索引移动到下一行时，矩阵是逐列存储的。这就是为什么Fortran被认为是**列优先语言**。另一方面，在C中，**最后一个**索引变化得最快。矩阵是按行存储的，使其成为**行优先语言**。你选择C还是Fortran取决于是否更重要地保留索引约定还是重新排序数据。

[在这里了解更多关于形状操作的信息](https://numpy.org/devdocs/user/quickstart.html#quickstart-shape-manipulation)

## 如何将1D数组转换为2D数组（如何给数组添加新轴）

*本节涵盖* `np.newaxis`，`np.expand_dims`

------------------------------------------------------------------------

你可以使用`np.newaxis`和`np.expand_dims`来增加现有数组的维度。

使用`np.newaxis`一次会增加数组的一个维度。这意味着**1D**数组将变成**2D**数组，**2D**数组将变成**3D**数组，以此类推。

例如，如果你从以下数组开始：

```python
>>> a = np.array([1, 2, 3, 4, 5, 6])
>>> a.shape
(6,)
```

你可以使用`np.newaxis`来添加一个新轴：

```python
>>> a2 = a[np.newaxis, :]
>>> a2.shape
(1, 6)
```

你可以使用`np.newaxis`明确地将1D数组转换为行向量或列向量。例如，你可以通过在第一个维度上插入一个轴来将1D数组转换为行向量：

```python
>>> row_vector = a[np.newaxis, :]
>>> row_vector.shape
(1, 6)
```

或者，如果你想得到一个列向量，你可以在第二个维度上插入一个轴：

```python
>>> col_vector = a[:, np.newaxis]
>>> col_vector.shape
(6, 1)
```

你还可以使用`np.expand_dims`在指定位置插入一个新轴来扩展数组。

例如，如果你从以下数组开始：

```python
>>> a = np.array([1, 2, 3, 4, 5, 6])
>>> a.shape
(6,)
```

你可以使用`np.expand_dims`在索引位置1处添加一个轴，如下：

```python
>>> b = np.expand_dims(a, axis=1)
>>> b.shape
(6, 1)
```

你可以在索引位置0处添加一个轴，如下：

```python
>>> c = np.expand_dims(a, axis=0)
>>> c.shape
(1, 6)
```

在[这里](https://numpy.org/devdocs/reference/routines.indexing.html#arrays-indexing)了解更多关于`newaxis`的信息，在[expand_dims](https://numpy.org/devdocs/reference/generated/numpy.expand_dims.html#numpy.expand_dims)页面了解更多关于`expand_dims`的信息。