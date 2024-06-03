# NumPy快速入门

## Prerequisites
You’ll need to know a bit of Python. For a refresher, see the [Python tutorial.](https://docs.python.org/tutorial/)

To work the examples, you’ll need `matplotlib` installed in addition to NumPy.

## 前提条件
你需要了解一些Python的基础知识。为了复习，请查阅[Python教程。](https://docs.python.org/tutorial/)

为了运行示例，除了NumPy外，你还需要安装`matplotlib`。

#### Learner profile
This is a quick overview of arrays in NumPy. It demonstrates how n-dimensional (
) arrays are represented and can be manipulated. In particular, if you don’t know how to apply common functions to n-dimensional arrays (without using for-loops), or if you want to understand axis and shape properties for n-dimensional arrays, this article might be of help.

#### 学习者画像
本文是对NumPy中数组的快速概述。它展示了n维数组是如何表示和操作的。特别是，如果你不知道如何对n维数组应用常见函数（不使用for循环），或者如果你想了解n维数组的轴和形状属性，本文可能会对你有所帮助。

#### Learning Objectives
After reading, you should be able to:

* Understand the difference between one-, two- and n-dimensional arrays in NumPy;

* Understand how to apply some linear algebra operations to n-dimensional arrays without using for-loops;

* Understand axis and shape properties for n-dimensional arrays.

#### 学习目标
阅读完本文后，你应该能够：

- 理解NumPy中一维、二维和n维数组之间的区别；
- 理解如何在不使用for循环的情况下对n维数组应用一些线性代数操作；
- 理解n维数组的轴和形状属性。

## The basics
NumPy’s main object is the homogeneous multidimensional array. It is a table of elements (usually numbers), all of the same type, indexed by a tuple of non-negative integers. In NumPy dimensions are called axes.

For example, the array for the coordinates of a point in 3D space, `[1, 2, 1]`, has one axis. That axis has 3 elements in it, so we say it has a length of 3. In the example pictured below, the array has 2 axes. The first axis has a length of 2, the second axis has a length of 3.

```
[[1., 0., 0.],
 [0., 1., 2.]]
```

NumPy’s array class is called `ndarray`. It is also known by the alias `array`. Note that `numpy.array` is not the same as the Standard Python Library class `array.array`, which only handles one-dimensional arrays and offers less functionality. The more important attributes of an `ndarray` object are:

**ndarray.ndim**
- the number of axes (dimensions) of the array.

**ndarray.shape**
- the dimensions of the array. This is a tuple of integers indicating the size of the array in each dimension. For a matrix with n rows and m columns, `shape` will be `(n,m)`. The length of the `shape` tuple is therefore the number of axes, `ndim`.

**ndarray.size**
- the total number of elements of the array. This is equal to the product of the elements of `shape`.

**ndarray.dtype**
- an object describing the type of the elements in the array. One can create or specify dtype’s using standard Python types. Additionally NumPy provides types of its own. numpy.int32, numpy.int16, and numpy.float64 are some examples.

**ndarray.itemsize**
- the size in bytes of each element of the array. For example, an array of elements of type `float64` has `itemsize` 8 (=64/8), while one of type `complex32` has `itemsize` 4 (=32/8). It is equivalent to `ndarray.dtype.itemsize`.

**ndarray.data**
- the buffer containing the actual elements of the array. Normally, we won’t need to use this attribute because we will access the elements in an array using indexing facilities.

## 基础知识
NumPy的主要对象是同质多维数组。它是一个元素表（通常是数字），所有元素类型相同，通过非负整数的元组进行索引。在NumPy中，维度被称为轴（axes）。

例如，表示三维空间中一个点坐标的数组`[1, 2, 1]`，它有一个轴。这个轴上有3个元素，所以我们说它的长度为3。在下面的示例图中，数组有2个轴。第一个轴的长度为2，第二个轴的长度为3。

```
[[1., 0., 0.],
 [0., 1., 2.]]
```

NumPy的数组类被称为`ndarray`，它也有别名`array`。请注意，`numpy.array`与标准Python库中的`array.array`类不同，后者仅处理一维数组且功能较少。`ndarray`对象的重要属性包括：

**ndarray.ndim**
- 数组的轴（维度）数量。

**ndarray.shape**
- 数组的维度。这是一个整数元组，表示数组在每个维度上的大小。对于具有n行和m列的矩阵，`shape`将是`(n,m)`。因此，`shape`元组的长度就是轴的数量，即`ndim`。

**ndarray.size**
- 数组元素的总数。这等于`shape`中元素的乘积。

**ndarray.dtype**
- 描述数组中元素类型的对象。可以使用标准Python类型创建或指定`dtype`。此外，NumPy还提供了自己的类型。例如，`numpy.int32`、`numpy.int16`和`numpy.float64`。

**ndarray.itemsize**
- 数组中每个元素的大小（以字节为单位）。例如，一个元素类型为`float64`的数组具有`itemsize`为8（=64/8），而一个元素类型为`complex32`的数组具有`itemsize`为4（=32/8）。它等同于`ndarray.dtype.itemsize`。

**ndarray.data**
- 包含数组实际元素的缓冲区。通常，我们不需要使用这个属性，因为我们将使用索引工具来访问数组中的元素。

### An example

```python
>>> import numpy as np
>>> a = np.arange(15).reshape(3, 5)
>>> a
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
>>> a.shape
(3, 5)
>>> a.ndim
2
>>> a.dtype.name
'int64'
>>> a.itemsize
8
>>> a.size
15
>>> type(a)
<class 'numpy.ndarray'>
>>> b = np.array([6, 7, 8])
>>> b
array([6, 7, 8])
>>> type(b)
<class 'numpy.ndarray'>
```

### 一个示例

```python
>>> import numpy as np
>>> a = np.arange(15).reshape(3, 5)
>>> a
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
>>> a.shape
(3, 5)
>>> a.ndim
2
>>> a.dtype.name
'int64'
>>> a.itemsize
8
>>> a.size
15
>>> type(a)
<class 'numpy.ndarray'>
>>> b = np.array([6, 7, 8])
>>> b
array([6, 7, 8])
>>> type(b)
<class 'numpy.ndarray'>
```

### Array creation
There are several ways to create arrays.

For example, you can create an array from a regular Python list or tuple using the `array` function. The type of the resulting array is deduced from the type of the elements in the sequences.

```python
>>> import numpy as np
>>> a = np.array([2, 3, 4])
>>> a
array([2, 3, 4])
>>> a.dtype
dtype('int64')
>>> b = np.array([1.2, 3.5, 5.1])
>>> b.dtype
dtype('float64')
```

A frequent error consists in calling array with multiple arguments, rather than providing a single sequence as an argument.

```python
>>> a = np.array(1, 2, 3, 4)    # WRONG
Traceback (most recent call last):
  ...
TypeError: array() takes from 1 to 2 positional arguments but 4 were given
>>> a = np.array([1, 2, 3, 4])  # RIGHT
```

`array` transforms sequences of sequences into two-dimensional arrays, sequences of sequences of sequences into three-dimensional arrays, and so on.

```python
>>> b = np.array([(1.5, 2, 3), (4, 5, 6)])
>>> b
array([[1.5, 2. , 3. ],
       [4. , 5. , 6. ]])
```

The type of the array can also be explicitly specified at creation time:

```python
>>> c = np.array([[1, 2], [3, 4]], dtype=complex)
>>> c
array([[1.+0.j, 2.+0.j],
       [3.+0.j, 4.+0.j]])
```

Often, the elements of an array are originally unknown, but its size is known. Hence, NumPy offers several functions to create arrays with initial placeholder content. These minimize the necessity of growing arrays, an expensive operation.

The function `zeros` creates an array full of zeros, the function `ones` creates an array full of ones, and the function `empty` creates an array whose initial content is random and depends on the state of the memory. By default, the dtype of the created array is `float64`, but it can be specified via the key word argument `dtype`.

```python
>>> np.zeros((3, 4))
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])
>>> np.ones((2, 3, 4), dtype=np.int16)
array([[[1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]],

       [[1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]]], dtype=int16)
>>> np.empty((2, 3)) 
array([[3.73603959e-262, 6.02658058e-154, 6.55490914e-260],  # may vary
       [5.30498948e-313, 3.14673309e-307, 1.00000000e+000]])
```

To create sequences of numbers, NumPy provides the `arange` function which is analogous to the Python built-in `range`, but returns an array.

```python
>>> np.arange(10, 30, 5)
array([10, 15, 20, 25])
>>> np.arange(0, 2, 0.3)  # it accepts float arguments
array([0. , 0.3, 0.6, 0.9, 1.2, 1.5, 1.8])
```

When `arange` is used with floating point arguments, it is generally not possible to predict the number of elements obtained, due to the finite floating point precision. For this reason, it is usually better to use the function `linspace` that receives as an argument the number of elements that we want, instead of the step:

```python
>>> from numpy import pi
>>> np.linspace(0, 2, 9)                   # 9 numbers from 0 to 2
array([0.  , 0.25, 0.5 , 0.75, 1.  , 1.25, 1.5 , 1.75, 2.  ])
>>> x = np.linspace(0, 2 * pi, 100)        # useful to evaluate function at lots of points
>>> f = np.sin(x)
```

### 数组创建
创建数组有多种方法。

例如，你可以使用`array`函数从普通的Python列表或元组创建一个数组。结果数组的类型是从序列中元素的类型推断出来的。

```python
>>> import numpy as np
>>> a = np.array([2, 3, 4])
>>> a
array([2, 3, 4])
>>> a.dtype
dtype('int64')
>>> b = np.array([1.2, 3.5, 5.1])
>>> b.dtype
dtype('float64')
```

一个常见的错误是调用`array`时提供了多个参数，而不是提供一个序列作为参数。

```python
>>> a = np.array(1, 2, 3, 4)    # 错误
Traceback (most recent call last):
  ...
TypeError: array() takes from 1 to 2 positional arguments but 4 were given
>>> a = np.array([1, 2, 3, 4])  # 正确
```

`array`将序列的序列转换为二维数组，将序列的序列的序列转换为三维数组，依此类推。

```python
>>> b = np.array([(1.5, 2, 3), (4, 5, 6)])
>>> b
array([[1.5, 2. , 3. ],
       [4. , 5. , 6. ]])
```

在创建时也可以明确指定数组的类型：

```python
>>> c = np.array([[1, 2], [3, 4]], dtype=complex)
>>> c
array([[1.+0.j, 2.+0.j],
       [3.+0.j, 4.+0.j]])
```

通常情况下，数组的元素最初是未知的，但其大小是已知的。因此，NumPy提供了几个函数来创建带有初始占位符内容的数组。这些函数尽量减少了数组扩展的需求，因为扩展数组是一个昂贵的操作。

`zeros` 函数创建一个全零数组，`ones` 函数创建一个全一数组，而 `empty` 函数创建一个初始内容为随机的数组，这些随机内容取决于内存的状态。默认情况下，创建的数组的数据类型（dtype）是 `float64`，但可以通过关键字参数 `dtype` 来指定。

```python
>>> np.zeros((3, 4))
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])
>>> np.ones((2, 3, 4), dtype=np.int16)
array([[[1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]],

       [[1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]]], dtype=int16)
>>> np.empty((2, 3)) 
array([[3.73603959e-262, 6.02658058e-154, 6.55490914e-260],  # 可能会变化
       [5.30498948e-313, 3.14673309e-307, 1.00000000e+000]])
```

为了创建数字序列，NumPy 提供了 `arange` 函数，该函数类似于 Python 内置的 `range` 函数，但返回的是一个数组。

```python
>>> np.arange(10, 30, 5)
array([10, 15, 20, 25])
>>> np.arange(0, 2, 0.3)  # 它接受浮点数参数
array([0. , 0.3, 0.6, 0.9, 1.2, 1.5, 1.8])
```

当使用浮点数参数调用 `arange` 时，由于有限的浮点精度，通常无法预测获得的元素数量。因此，通常更好的做法是使用 `linspace` 函数，它接收我们想要的元素数量作为参数，而不是步长：

```python
>>> from numpy import pi
>>> np.linspace(0, 2, 9)                   # 从0到2的9个数
array([0.  , 0.25, 0.5 , 0.75, 1.  , 1.25, 1.5 , 1.75, 2.  ])
>>> x = np.linspace(0, 2 * pi, 100)        # 在大量点上评估函数时很有用
>>> f = np.sin(x)
```

### Printing arrays
When you print an array, NumPy displays it in a similar way to nested lists, but with the following layout:

* the last axis is printed from left to right,
* the second-to-last is printed from top to bottom,
* the rest are also printed from top to bottom, with each slice separated from the next by an empty line.

One-dimensional arrays are then printed as rows, bidimensionals as matrices and tridimensionals as lists of matrices.

```python
>>> a = np.arange(6)                    # 1d array
>>> print(a)
[0 1 2 3 4 5]

>>> b = np.arange(12).reshape(4, 3)     # 2d array
>>> print(b)
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]

>>> c = np.arange(24).reshape(2, 3, 4)  # 3d array
>>> print(c)
[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]

 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]]
```

See below to get more details on `reshape`.

If an array is too large to be printed, NumPy automatically skips the central part of the array and only prints the corners:

```python
>>> print(np.arange(10000))
[   0    1    2 ... 9997 9998 9999]

>>> print(np.arange(10000).reshape(100, 100))
[[   0    1    2 ...   97   98   99]
 [ 100  101  102 ...  197  198  199]
 [ 200  201  202 ...  297  298  299]
 ...
 [9700 9701 9702 ... 9797 9798 9799]
 [9800 9801 9802 ... 9897 9898 9899]
 [9900 9901 9902 ... 9997 9998 9999]]
```

To disable this behaviour and force NumPy to print the entire array, you can change the printing options using `set_printoptions`.

```python
>>> np.set_printoptions(threshold=sys.maxsize)  # sys module should be imported
```

### 打印数组
当你打印一个数组时，NumPy 会以类似于嵌套列表的方式显示它，但布局如下：

* 最后一个轴从左到右打印，
* 倒数第二个轴从上到下打印，
* 其余轴也是从上到下打印，每个切片之间用空行隔开。

一维数组作为行打印，二维数组作为矩阵打印，三维数组作为矩阵列表打印。

```python
>>> a = np.arange(6)                    # 一维数组
>>> print(a)
[0 1 2 3 4 5]

>>> b = np.arange(12).reshape(4, 3)     # 二维数组
>>> print(b)
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]

>>> c = np.arange(24).reshape(2, 3, 4)  # 三维数组
>>> print(c)
[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]

 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]]
```

关于 `reshape` 的更多细节请见下文。

如果数组太大而无法打印，NumPy 会自动跳过数组的中间部分，只打印角落部分：

```python
>>> print(np.arange(10000))
[   0    1    2 ... 9997 9998 9999]

>>> print(np.arange(10000).reshape(100, 100))
[[   0    1    2 ...   97   98   99]
 [ 100  101  102 ...  197  198  199]
 [ 200  201  202 ...  297  298  299]
 ...
 [9700 9701 9702 ... 9797 9798 9799]
 [9800 9801 9802 ... 9897 9898 9899]
 [9900 9901 9902 ... 9997 9998 9999]]
```

要禁用这种行为并强制 NumPy 打印整个数组，你可以使用 `set_printoptions` 来更改打印选项。

```python
>>> np.set_printoptions(threshold=sys.maxsize)  # 应先导入 sys 模块
```

### Basic operations
Arithmetic operators on arrays apply elementwise. A new array is created and filled with the result.

```python
>>> a = np.array([20, 30, 40, 50])
>>> b = np.arange(4)
>>> b
array([0, 1, 2, 3])
>>> c = a - b
>>> c
array([20, 29, 38, 47])
>>> b**2
array([0, 1, 4, 9])
>>> 10 * np.sin(a)
array([ 9.12945251, -9.88031624,  7.4511316 , -2.62374854])
>>> a < 35
array([ True,  True, False, False])
```

Unlike in many matrix languages, the product operator `*` operates elementwise in NumPy arrays. The matrix product can be performed using the `@` operator (in python >=3.5) or the dot function or method:

```python
>>> A = np.array([[1, 1],
              [0, 1]])
>>> B = np.array([[2, 0],
              [3, 4]])
>>> A * B     # elementwise product
array([[2, 0],
       [0, 4]])
>>> A @ B     # matrix product
array([[5, 4],
       [3, 4]])
>>> A.dot(B)  # another matrix product
array([[5, 4],
       [3, 4]])
```

Some operations, such as `+=` and `*=`, act in place to modify an existing array rather than create a new one.

```python
>>> rg = np.random.default_rng(1)  # create instance of default random number generator
>>> a = np.ones((2, 3), dtype=int)
>>> b = rg.random((2, 3))
>>> a *= 3
>>> a
array([[3, 3, 3],
       [3, 3, 3]])
>>> b += a
>>> b
array([[3.51182162, 3.9504637 , 3.14415961],
       [3.94864945, 3.31183145, 3.42332645]])
>>> a += b  # b is not automatically converted to integer type
Traceback (most recent call last):
    ...
numpy._core._exceptions._UFuncOutputCastingError: Cannot cast ufunc 'add' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'
```

When operating with arrays of different types, the type of the resulting array corresponds to the more general or precise one (a behavior known as upcasting).

```python
>>> a = np.ones(3, dtype=np.int32)
>>> b = np.linspace(0, pi, 3)
>>> b.dtype.name
'float64'
>>> c = a + b
>>> c
array([1.        , 2.57079633, 4.14159265])
>>> c.dtype.name
'float64'
>>> d = np.exp(c * 1j)
>>> d
array([ 0.54030231+0.84147098j, -0.84147098+0.54030231j,
       -0.54030231-0.84147098j])
>>> d.dtype.name
'complex128'
```

Many unary operations, such as computing the sum of all the elements in the array, are implemented as methods of the `ndarray` class.

```python
>>> a = rg.random((2, 3))
>>> a
array([[0.82770259, 0.40919914, 0.54959369],
       [0.02755911, 0.75351311, 0.53814331]])
>>> a.sum()
3.1057109529998157
>>> a.min()
0.027559113243068367
>>> a.max()
0.8277025938204418
```

By default, these operations apply to the array as though it were a list of numbers, regardless of its shape. However, by specifying the `axis` parameter you can apply an operation along the specified axis of an array:

```python
>>> b = np.arange(12).reshape(3, 4)
>>> b
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])

>>> b.sum(axis=0)     # sum of each column
array([12, 15, 18, 21])

>>> b.min(axis=1)     # min of each row
array([0, 4, 8])

>>> b.cumsum(axis=1)  # cumulative sum along each row
array([[ 0,  1,  3,  6],
       [ 4,  9, 15, 22],
       [ 8, 17, 27, 38]])
```

### 基本操作
数组上的算术运算符是逐个元素应用的。会创建一个新数组并用结果填充。

```python
>>> a = np.array([20, 30, 40, 50])
>>> b = np.arange(4)
>>> b
array([0, 1, 2, 3])
>>> c = a - b
>>> c
array([20, 29, 38, 47])
>>> b**2
array([0, 1, 4, 9])
>>> 10 * np.sin(a)
array([ 9.12945251, -9.88031624,  7.4511316 , -2.62374854])
>>> a < 35
array([ True,  True, False, False])
```

与许多矩阵语言不同，NumPy 数组中的乘法运算符 `*` 是逐个元素进行的。矩阵乘积可以使用 `@` 运算符（在 Python 3.5 及以上版本中）或者 dot 函数或方法来实现：

```python
>>> A = np.array([[1, 1],
                  [0, 1]])
>>> B = np.array([[2, 0],
                  [3, 4]])
>>> A * B     # 逐个元素乘积
array([[2, 0],
       [0, 4]])
>>> A @ B     # 矩阵乘积
array([[5, 4],
       [3, 4]])
>>> A.dot(B)  # 另一种矩阵乘积
array([[5, 4],
       [3, 4]])
```

一些操作，如 `+=` 和 `*=`，会就地修改现有数组而不是创建新的数组。

```python
>>> rg = np.random.default_rng(1)  # 创建默认随机数生成器的实例
>>> a = np.ones((2, 3), dtype=int)
>>> b = rg.random((2, 3))
>>> a *= 3
>>> a
array([[3, 3, 3],
       [3, 3, 3]])
>>> b += a
>>> b
array([[3.51182162, 3.9504637 , 3.14415961],
       [3.94864945, 3.31183145, 3.42332645]])
>>> a += b  # b 不会自动转换为整数类型
Traceback (most recent call last):
    ...
numpy._core._exceptions._UFuncOutputCastingError: 无法使用转换规则 'same_kind' 将 ufunc 'add' 的输出从 dtype('float64') 转换为 dtype('int64')
```

当使用不同类型的数组进行操作时，结果数组的类型对应于更通用或更精确的类型（这种行为被称为向上转型）。

```python
>>> a = np.ones(3, dtype=np.int32)
>>> b = np.linspace(0, pi, 3)
>>> b.dtype.name
'float64'
>>> c = a + b
>>> c
array([1.        , 2.57079633, 4.14159265])
>>> c.dtype.name
'float64'
>>> d = np.exp(c * 1j)
>>> d
array([ 0.54030231+0.84147098j, -0.84147098+0.54030231j,
       -0.54030231-0.84147098j])
>>> d.dtype.name
'complex128'
```

许多一元操作，如计算数组中所有元素的和，都被实现为 `ndarray` 类的方法。

```python
>>> a = rg.random((2, 3))
>>> a
array([[0.82770259, 0.40919914, 0.54959369],
       [0.02755911, 0.75351311, 0.53814331]])
>>> a.sum()
3.1057109529998157
>>> a.min()
0.027559113243068367
>>> a.max()
0.8277025938204418
```

默认情况下，这些操作会应用于数组，就像它是一个数字列表一样，不考虑其形状。但是，通过指定 `axis` 参数，你可以沿着数组的指定轴应用一个操作：

```python
>>> b = np.arange(12).reshape(3, 4)
>>> b
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])

>>> b.sum(axis=0)     # 每列的和
array([12, 15, 18, 21])

>>> b.min(axis=1)     # 每行的最小值
array([0, 4, 8])

>>> b.cumsum(axis=1)  # 沿着每行的累积和
array([[ 0,  1,  3,  6],
       [ 4,  9, 15, 22],
       [ 8, 17, 27, 38]])
```

### Universal functions
NumPy provides familiar mathematical functions such as sin, cos, and exp. In NumPy, these are called “universal functions” (`ufunc`). Within NumPy, these functions operate elementwise on an array, producing an array as output.

```python
>>> B = np.arange(3)
>>> B
array([0, 1, 2])
>>> np.exp(B)
array([1.        , 2.71828183, 7.3890561 ])
>>> np.sqrt(B)
array([0.        , 1.        , 1.41421356])
>>> C = np.array([2., -1., 4.])
>>> np.add(B, C)
array([2., 0., 6.])
```

### 通用函数
NumPy 提供了熟悉的数学函数，如 sin、cos 和 exp。在 NumPy 中，这些函数被称为“通用函数”（`ufunc`）。在 NumPy 中，这些函数会对数组中的每个元素进行逐元素操作，并生成一个数组作为输出。

```python
>>> B = np.arange(3)
>>> B
array([0, 1, 2])
>>> np.exp(B)
array([1.        , 2.71828183, 7.3890561 ])
# 这里对数组 B 中的每个元素计算了 e 的指数幂

>>> np.sqrt(B)
array([0.        , 1.        , 1.41421356])
# 这里对数组 B 中的每个元素计算了平方根

>>> C = np.array([2., -1., 4.])
>>> np.add(B, C)
array([2., 0., 6.])
# 这里使用了 np.add 函数对数组 B 和 C 中的元素进行了逐元素相加
```

### Indexing, slicing and iterating
**One-dimensional** arrays can be indexed, sliced and iterated over, much like lists and other Python sequences.

```python
>>> a = np.arange(10)**3
>>> a
array([  0,   1,   8,  27,  64, 125, 216, 343, 512, 729])
>>> a[2]
8
>>> a[2:5]
array([ 8, 27, 64])
>>> # equivalent to a[0:6:2] = 1000;
>>> # from start to position 6, exclusive, set every 2nd element to 1000
>>> a[:6:2] = 1000
>>> a
array([1000,    1, 1000,   27, 1000,  125,  216,  343,  512,  729])
>>> a[::-1]  # reversed a
array([ 729,  512,  343,  216,  125, 1000,   27, 1000,    1, 1000])
>>> for i in a:
    print(i**(1 / 3.))

9.999999999999998  # may vary
1.0
9.999999999999998
3.0
9.999999999999998
4.999999999999999
5.999999999999999
6.999999999999999
7.999999999999999
8.999999999999998
```

**Multidimensional** arrays can have one index per axis. These indices are given in a tuple separated by commas:

```python
>>> def f(x, y):
    return 10 * x + y

>>> b = np.fromfunction(f, (5, 4), dtype=int)
>>> b
array([[ 0,  1,  2,  3],
       [10, 11, 12, 13],
       [20, 21, 22, 23],
       [30, 31, 32, 33],
       [40, 41, 42, 43]])
>>> b[2, 3]
23
>>> b[0:5, 1]  # each row in the second column of b
array([ 1, 11, 21, 31, 41])
>>> b[:, 1]    # equivalent to the previous example
array([ 1, 11, 21, 31, 41])
>>> b[1:3, :]  # each column in the second and third row of b
array([[10, 11, 12, 13],
       [20, 21, 22, 23]])
```

When fewer indices are provided than the number of axes, the missing indices are considered complete slices `:`

```python
>>> b[-1]   # the last row. Equivalent to b[-1, :]
array([40, 41, 42, 43])
```

The expression within brackets in `b[i]` is treated as an `i` followed by as many instances of `:` as needed to represent the remaining axes. NumPy also allows you to write this using dots as `b[i, ...]`.

The **dots** (`...`) represent as many colons as needed to produce a complete indexing tuple. For example, if `x` is an array with 5 axes, then
`
* `x[1, 2, ...]` is equivalent to `x[1, 2, :, :, :]`,
* `x[..., 3]` to `x[:, :, :, :, 3]` and
* `x[4, ..., 5, :]` to `x[4, :, :, 5, :]`.

```python
>>> c = np.array([[[  0,  1,  2],  # a 3D array (two stacked 2D arrays)
               [ 10, 12, 13]],
              [[100, 101, 102],
               [110, 112, 113]]])
>>> c.shape
(2, 2, 3)
>>> c[1, ...]  # same as c[1, :, :] or c[1]
array([[100, 101, 102],
       [110, 112, 113]])
>>> c[..., 2]  # same as c[:, :, 2]
array([[  2,  13],
       [102, 113]])
```

**Iterating** over multidimensional arrays is done with respect to the first axis:

```python
for row in b:
    print(row)

[0 1 2 3]
[10 11 12 13]
[20 21 22 23]
[30 31 32 33]
[40 41 42 43]
```

However, if one wants to perform an operation on each element in the array, one can use the `flat` attribute which is an iterator over all the elements of the array:

```python
>>> for element in b.flat:
    print(element)

0
1
2
3
10
11
12
13
20
21
22
23
30
31
32
33
40
41
42
43
```

### 索引、切片和迭代

**一维**数组可以被索引、切片和迭代，就像列表和其他Python序列一样。

```python
>>> a = np.arange(10)**3
>>> a
array([  0,   1,   8,  27,  64, 125, 216, 343, 512, 729])
>>> a[2]
8
>>> a[2:5]
array([ 8, 27, 64])
>>> # 相当于 a[0:6:2] = 1000;
>>> # 从开始到位置6（不包括），将每两个元素设置为1000
>>> a[:6:2] = 1000
>>> a
array([1000,    1, 1000,   27, 1000,  125,  216,  343,  512,  729])
>>> a[::-1]  # 反转数组a
array([ 729,  512,  343,  216,  125, 1000,   27, 1000,    1, 1000])
>>> for i in a:
    print(i**(1 / 3.))

9.999999999999998  # 可能有所变化
1.0
9.999999999999998
3.0
9.999999999999998
4.999999999999999
5.999999999999999
6.999999999999999
7.999999999999999
8.999999999999998
```

**多维**数组的每个轴都可以有一个索引。这些索引以逗号分隔的元组形式给出：

```python
>>> def f(x, y):
    return 10 * x + y

>>> b = np.fromfunction(f, (5, 4), dtype=int)
>>> b
array([[ 0,  1,  2,  3],
       [10, 11, 12, 13],
       [20, 21, 22, 23],
       [30, 31, 32, 33],
       [40, 41, 42, 43]])
>>> b[2, 3]
23
>>> b[0:5, 1]  # 数组b的第二列中的每一行
array([ 1, 11, 21, 31, 41])
>>> b[:, 1]    # 与前一个示例等效
array([ 1, 11, 21, 31, 41])
>>> b[1:3, :]  # 数组b的第二行和第三行中的每一列
array([[10, 11, 12, 13],
       [20, 21, 22, 23]])
```

当提供的索引数少于轴数时，缺失的索引被视为完整的切片`:`。

```python
>>> b[-1]   # 最后一行。相当于b[-1, :]
array([40, 41, 42, 43])
```

在`b[i]`中，括号内的表达式被视为`i`后跟足够数量的`:`，以表示剩余的轴。NumPy也允许你使用点号来表示`b[i, ...]`。

**点号**（`...`）代表足够数量的冒号以生成一个完整的索引元组。例如，如果`x`是一个有5个轴的数组，那么：

* `x[1, 2, ...]` 相当于 `x[1, 2, :, :, :]`，
* `x[..., 3]` 相当于 `x[:, :, :, :, 3]`，
* `x[4, ..., 5, :]` 相当于 `x[4, :, :, 5, :]`。

```python
>>> c = np.array([[[  0,  1,  2],  # 一个3D数组（两个堆叠的2D数组）
               [ 10, 12, 13]],
              [[100, 101, 102],
               [110, 112, 113]]])
>>> c.shape
(2, 2, 3)
>>> c[1, ...]  # 与 c[1, :, :] 或 c[1] 等效
array([[100, 101, 102],
       [110, 112, 113]])
>>> c[..., 2]  # 与 c[:, :, 2] 等效
array([[  2,  13],
       [102, 113]])
```

**迭代**多维数组时，是相对于第一个轴进行的：

```python
for row in b:
    print(row)

[0 1 2 3]
[10 11 12 13]
[20 21 22 23]
[30 31 32 33]
[40 41 42 43]
```

然而，如果想对数组中的每个元素执行操作，可以使用`flat`属性，它是一个遍历数组所有元素的迭代器：

```python
>>> for element in b.flat:
    print(element)

0
1
2
3
10
11
12
13
20
21
22
23
30
31
32
33
40
41
42
43
```

## Shape manipulation

### Changing the shape of an array
An array has a shape given by the number of elements along each axis:
```python
>>> a = np.floor(10 * rg.random((3, 4)))
>>> a
array([[3., 7., 3., 4.],
       [1., 4., 2., 2.],
       [7., 2., 4., 9.]])
>>> a.shape
(3, 4)
```

The shape of an array can be changed with various commands. Note that the following three commands all return a modified array, but do not change the original array:

```python
>>> a.ravel()  # returns the array, flattened
array([3., 7., 3., 4., 1., 4., 2., 2., 7., 2., 4., 9.])
>>> a.reshape(6, 2)  # returns the array with a modified shape
array([[3., 7.],
       [3., 4.],
       [1., 4.],
       [2., 2.],
       [7., 2.],
       [4., 9.]])
>>> a.T  # returns the array, transposed
array([[3., 1., 7.],
       [7., 4., 2.],
       [3., 2., 4.],
       [4., 2., 9.]])
>>> a.T.shape
(4, 3)
>>> a.shape
(3, 4)
```

The order of the elements in the array resulting from `ravel` is normally “C-style”, that is, the rightmost index “changes the fastest”, so the element after `a[0, 0]` is `a[0, 1]`. If the array is reshaped to some other shape, again the array is treated as “C-style”. NumPy normally creates arrays stored in this order, so `ravel` will usually not need to copy its argument, but if the array was made by taking slices of another array or created with unusual options, it may need to be copied. The functions `ravel` and `reshape` can also be instructed, using an optional argument, to use FORTRAN-style arrays, in which the leftmost index changes the fastest.

The [reshape](https://numpy.org/devdocs/reference/generated/numpy.reshape.html#numpy.reshape) function returns its argument with a modified shape, whereas the [ndarray.resize](https://numpy.org/devdocs/reference/generated/numpy.ndarray.resize.html#numpy.ndarray.resize) method modifies the array itself:

```python
>>> a
array([[3., 7., 3., 4.],
       [1., 4., 2., 2.],
       [7., 2., 4., 9.]])
>>> a.resize((2, 6))
>>> a
array([[3., 7., 3., 4., 1., 4.],
       [2., 2., 7., 2., 4., 9.]])
```

If a dimension is given as `-1` in a reshaping operation, the other dimensions are automatically calculated:

```python
>>> a.reshape(3, -1)
array([[3., 7., 3., 4.],
       [1., 4., 2., 2.],
       [7., 2., 4., 9.]])
```

## 形状操作

### 改变数组的形状
数组的形状由它每个维度（或称为轴）上的元素数量确定：

```python
>>> a = np.floor(10 * rg.random((3, 4)))
>>> a
array([[3., 7., 3., 4.],
       [1., 4., 2., 2.],
       [7., 2., 4., 9.]])
>>> a.shape
(3, 4)
```

我们可以使用各种命令来改变数组的形状。但请注意，以下三个命令都会返回一个新形状的数组，而不会修改原始数组：

```python
>>> a.ravel()  # 将数组展平为一维数组
array([3., 7., 3., 4., 1., 4., 2., 2., 7., 2., 4., 9.])

>>> a.reshape(6, 2)  # 改变数组的形状为6行2列
array([[3., 7.],
       [3., 4.],
       [1., 4.],
       [2., 2.],
       [7., 2.],
       [4., 9.]])

>>> a.T  # 将数组进行转置，即行变列，列变行
array([[3., 1., 7.],
       [7., 4., 2.],
       [3., 2., 4.],
       [4., 2., 9.]])

>>> a.T.shape
(4, 3)

>>> a.shape
(3, 4)
```

`ravel` 生成的数组元素的顺序通常是“C风格”的，即最右边的索引“变化最快”，所以 `a[0, 0]` 之后的元素是 `a[0, 1]`。如果数组被重塑为其他形状，同样会将数组视为“C风格”的。NumPy 通常以这种顺序存储数组，因此 `ravel` 通常不需要复制其参数，但如果数组是通过另一个数组的切片或使用非标准选项创建的，则可能需要复制。通过使用可选参数，`ravel` 和 `reshape` 函数也可以被指示使用 FORTRAN 风格的数组，其中最左边的索引变化最快。

[reshape](https://numpy.org/devdocs/reference/generated/numpy.reshape.html#numpy.reshape) 函数返回具有修改后形状的参数数组，而 [ndarray.resize](https://numpy.org/devdocs/reference/generated/numpy.ndarray.resize.html#numpy.ndarray.resize) 方法则修改数组本身：

```python
>>> a
array([[3., 7., 3., 4.],
       [1., 4., 2., 2.],
       [7., 2., 4., 9.]])
>>> a.resize((2, 6))
>>> a
array([[3., 7., 3., 4., 1., 4.],
       [2., 2., 7., 2., 4., 9.]])
```

如果在重塑操作中某个维度被指定为 `-1`，则其他维度将自动计算：

```python
>>> a.reshape(3, -1)
array([[3., 7., 3., 4.],
       [1., 4., 2., 2.],
       [7., 2., 4., 9.]])
```

### Stacking together different arrays
Several arrays can be stacked together along different axes:

```python
>>> a = np.floor(10 * rg.random((2, 2)))
>>> a
array([[9., 7.],
       [5., 2.]])
>>> b = np.floor(10 * rg.random((2, 2)))
>>> b
array([[1., 9.],
       [5., 1.]])
>>> np.vstack((a, b))
array([[9., 7.],
       [5., 2.],
       [1., 9.],
       [5., 1.]])
>>> np.hstack((a, b))
array([[9., 7., 1., 9.],
       [5., 2., 5., 1.]])
```

The function [column_stack](https://numpy.org/devdocs/reference/generated/numpy.column_stack.html#numpy.column_stack) stacks 1D arrays as columns into a 2D array. It is equivalent to [hstack](https://numpy.org/devdocs/reference/generated/numpy.hstack.html#numpy.hstack) only for 2D arrays:

```python
>>> from numpy import newaxis
>>> np.column_stack((a, b))  # with 2D arrays
array([[9., 7., 1., 9.],
       [5., 2., 5., 1.]])
>>> a = np.array([4., 2.])
>>> b = np.array([3., 8.])
>>> np.column_stack((a, b))  # returns a 2D array
array([[4., 3.],
       [2., 8.]])
>>> np.hstack((a, b))        # the result is different
array([4., 2., 3., 8.])
>>> a[:, newaxis]  # view `a` as a 2D column vector
array([[4.],
       [2.]])
>>> np.column_stack((a[:, newaxis], b[:, newaxis]))
array([[4., 3.],
       [2., 8.]])
>>> np.hstack((a[:, newaxis], b[:, newaxis]))  # the result is the same
array([[4., 3.],
       [2., 8.]])
```

In general, for arrays with more than two dimensions, [hstack](https://numpy.org/devdocs/reference/generated/numpy.hstack.html#numpy.hstack) stacks along their second axes, [vstack](https://numpy.org/devdocs/reference/generated/numpy.vstack.html#numpy.vstack) stacks along their first axes, and [concatenate](https://numpy.org/devdocs/reference/generated/numpy.concatenate.html#numpy.concatenate) allows for an optional arguments giving the number of the axis along which the concatenation should happen.

#### Note

In complex cases, [r_](https://numpy.org/devdocs/reference/generated/numpy.r_.html#numpy.r_) and [c_](https://numpy.org/devdocs/reference/generated/numpy.c_.html#numpy.c_) are useful for creating arrays by stacking numbers along one axis. They allow the use of range literals `:`.

```python
>>> np.r_[1:4, 0, 4]
array([1, 2, 3, 0, 4])
```

When used with arrays as arguments, `r_` and `c_` are similar to `vstack` and `hstack` in their default behavior, but allow for an optional argument giving the number of the axis along which to concatenate.

### 将不同的数组堆叠在一起
几个数组可以沿着不同的轴堆叠在一起：

```python
>>> a = np.floor(10 * rg.random((2, 2)))
>>> a
array([[9., 7.],
       [5., 2.]])
>>> b = np.floor(10 * rg.random((2, 2)))
>>> b
array([[1., 9.],
       [5., 1.]])
>>> np.vstack((a, b))
array([[9., 7.],
       [5., 2.],
       [1., 9.],
       [5., 1.]])
>>> np.hstack((a, b))
array([[9., 7., 1., 9.],
       [5., 2., 5., 1.]])
```

函数 [column_stack](https://numpy.org/devdocs/reference/generated/numpy.column_stack.html#numpy.column_stack) 将一维数组作为列堆叠到二维数组中。对于二维数组来说，它等同于 [hstack](https://numpy.org/devdocs/reference/generated/numpy.hstack.html#numpy.hstack)：

```python
>>> from numpy import newaxis
>>> np.column_stack((a, b))  # 对于二维数组
array([[9., 7., 1., 9.],
       [5., 2., 5., 1.]])
>>> a = np.array([4., 2.])
>>> b = np.array([3., 8.])
>>> np.column_stack((a, b))  # 返回一个二维数组
array([[4., 3.],
       [2., 8.]])
>>> np.hstack((a, b))        # 结果不同
array([4., 2., 3., 8.])
>>> a[:, newaxis]  # 将 `a` 看作一个二维列向量
array([[4.],
       [2.]])
>>> np.column_stack((a[:, newaxis], b[:, newaxis]))
array([[4., 3.],
       [2., 8.]])
>>> np.hstack((a[:, newaxis], b[:, newaxis]))  # 结果相同
array([[4., 3.],
       [2., 8.]])
```

一般来说，对于超过两个维度的数组，[hstack](https://numpy.org/devdocs/reference/generated/numpy.hstack.html#numpy.hstack) 是沿着它们的第二个轴进行堆叠的，[vstack](https://numpy.org/devdocs/reference/generated/numpy.vstack.html#numpy.vstack) 是沿着它们的第一个轴进行堆叠的，而 [concatenate](https://numpy.org/devdocs/reference/generated/numpy.concatenate.html#numpy.concatenate) 允许一个可选参数来指定应该沿着哪个轴进行连接。

#### 注意

在复杂的情况下，[r_](https://numpy.org/devdocs/reference/generated/numpy.r_.html#numpy.r_) 和 [c_](https://numpy.org/devdocs/reference/generated/numpy.c_.html#numpy.c_) 对于通过沿着一个轴堆叠数字来创建数组是非常有用的。它们允许使用范围字面量 `:`。

```python
>>> np.r_[1:4, 0, 4]
array([1, 2, 3, 0, 4])
```

当使用数组作为参数时，`r_` 和 `c_` 在默认行为上与 `vstack` 和 `hstack` 类似，但允许一个可选参数来指定要连接的轴号。

### Splitting one array into several smaller ones
Using [hsplit](https://numpy.org/devdocs/reference/generated/numpy.hsplit.html#numpy.hsplit), you can split an array along its horizontal axis, either by specifying the number of equally shaped arrays to return, or by specifying the columns after which the division should occur:

```python
>>> a = np.floor(10 * rg.random((2, 12)))
>>> a
array([[6., 7., 6., 9., 0., 5., 4., 0., 6., 8., 5., 2.],
       [8., 5., 5., 7., 1., 8., 6., 7., 1., 8., 1., 0.]])
# Split `a` into 3
>>> np.hsplit(a, 3)
[array([[6., 7., 6., 9.],
       [8., 5., 5., 7.]]), array([[0., 5., 4., 0.],
       [1., 8., 6., 7.]]), array([[6., 8., 5., 2.],
       [1., 8., 1., 0.]])]
# Split `a` after the third and the fourth column
>>> np.hsplit(a, (3, 4))
[array([[6., 7., 6.],
       [8., 5., 5.]]), array([[9.],
       [7.]]), array([[0., 5., 4., 0., 6., 8., 5., 2.],
       [1., 8., 6., 7., 1., 8., 1., 0.]])]
```

[vsplit](https://numpy.org/devdocs/reference/generated/numpy.vsplit.html#numpy.vsplit) splits along the vertical axis, and [array_split](https://numpy.org/devdocs/reference/generated/numpy.array_split.html#numpy.array_split) allows one to specify along which axis to split.

### 将一个数组分割为多个较小的数组
使用 [hsplit](https://numpy.org/devdocs/reference/generated/numpy.hsplit.html#numpy.hsplit)，你可以沿着数组的水平轴进行分割，既可以指定要返回的具有相同形状的数组的数量，也可以指定分割应发生的列之后的位置：

```python
>>> a = np.floor(10 * rg.random((2, 12)))
>>> a
array([[6., 7., 6., 9., 0., 5., 4., 0., 6., 8., 5., 2.],
       [8., 5., 5., 7., 1., 8., 6., 7., 1., 8., 1., 0.]])
# 将 `a` 分割为3个数组
>>> np.hsplit(a, 3)
[array([[6., 7., 6., 9.],
       [8., 5., 5., 7.]]), array([[0., 5., 4., 0.],
       [1., 8., 6., 7.]]), array([[6., 8., 5., 2.],
       [1., 8., 1., 0.]])]
# 在第三列和第四列之后分割 `a`
>>> np.hsplit(a, (3, 4))
[array([[6., 7., 6.],
       [8., 5., 5.]]), array([[9.],
       [7.]]), array([[0., 5., 4., 0., 6., 8., 5., 2.],
       [1., 8., 6., 7., 1., 8., 1., 0.]])]
```

[vsplit](https://numpy.org/devdocs/reference/generated/numpy.vsplit.html#numpy.vsplit) 是沿着垂直轴进行分割的，而 [array_split](https://numpy.org/devdocs/reference/generated/numpy.array_split.html#numpy.array_split) 则允许你指定要沿着哪个轴进行分割。

## Copies and views
When operating and manipulating arrays, their data is sometimes copied into a new array and sometimes not. This is often a source of confusion for beginners. There are three cases:

## 副本和视图
在操作和修改数组时，有时数据会被复制到新的数组中，有时则不会。这常常会让初学者感到困惑。这里有三种情况：

### No copy at all
Simple assignments make no copy of objects or their data.

```python
>>> a = np.array([[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]])
>>> b = a            # no new object is created
>>> b is a           # a and b are two names for the same ndarray object
True
```

Python passes mutable objects as references, so function calls make no copy.

```python
>>> def f(x):
    print(id(x))

>>> id(a)  # id is a unique identifier of an object 
148293216  # may vary
>>> f(a)   
148293216  # may vary
```

### 完全不复制
简单的赋值操作不会复制对象或其数据。

```python
>>> a = np.array([[ 0,  1,  2,  3],
                  [ 4,  5,  6,  7],
                  [ 8,  9, 10, 11]])
>>> b = a            # 没有创建新的对象
>>> b is a           # a 和 b 是指向同一个 ndarray 对象的两个名称
True
```

Python 将可变对象作为引用传递，因此函数调用不会进行复制。

```python
>>> def f(x):
    print(id(x))

>>> id(a)  # id 是对象的唯一标识符
148293216  # 可能会变
>>> f(a)   
148293216  # 可能会变
```

### View or shallow copy
Different array objects can share the same data. The `view` method creates a new array object that looks at the same data.

```python
>>> c = a.view()
>>> c is a
False
>>> c.base is a            # c is a view of the data owned by a
True
>>> c.flags.owndata
False

>>> c = c.reshape((2, 6))  # a's shape doesn't change
>>> a.shape
(3, 4)
>>> c[0, 4] = 1234         # a's data changes
>>> a
array([[   0,    1,    2,    3],
       [1234,    5,    6,    7],
       [   8,    9,   10,   11]])
```

Slicing an array returns a view of it:

```python
>>> s = a[:, 1:3]
>>> s[:] = 10  # s[:] is a view of s. Note the difference between s = 10 and s[:] = 10
>>> a
array([[   0,   10,   10,    3],
       [1234,   10,   10,    7],
       [   8,   10,   10,   11]])
```

### 视图或浅拷贝
不同的数组对象可以共享相同的数据。`view` 方法会创建一个新的数组对象，该对象查看相同的数据。

```python
>>> c = a.view()
>>> c is a
False
>>> c.base is a            # c 是 a 所拥有数据的视图
True
>>> c.flags.owndata
False

>>> c = c.reshape((2, 6))  # a 的形状不会改变
>>> a.shape
(3, 4)
>>> c[0, 4] = 1234         # a 的数据发生改变
>>> a
array([[   0,    1,    2,    3],
       [1234,    5,    6,    7],
       [   8,    9,   10,   11]])
```

对数组进行切片操作会返回它的一个视图：

```python
>>> s = a[:, 1:3]
>>> s[:] = 10  # s[:] 是 s 的视图。注意 s = 10 和 s[:] = 10 之间的区别
>>> a
array([[   0,   10,   10,    3],
       [1234,   10,   10,    7],
       [   8,   10,   10,   11]])
```

### Deep copy
The `copy` method makes a complete copy of the array and its data.

```python
>>> d = a.copy()  # a new array object with new data is created
>>> d is a
False
>>> d.base is a  # d doesn't share anything with a
False
>>> d[0, 0] = 9999
>>> a
array([[   0,   10,   10,    3],
       [1234,   10,   10,    7],
       [   8,   10,   10,   11]])
```

Sometimes `copy` should be called after slicing if the original array is not required anymore. For example, suppose `a` is a huge intermediate result and the final result `b` only contains a small fraction of `a`, a deep copy should be made when constructing `b` with slicing:

```python
>>> a = np.arange(int(1e8))
>>> b = a[:100].copy()
>>> del a  # the memory of ``a`` can be released.
```

If `b = a[:100]` is used instead, `a` is referenced by `b` and will persist in memory even if `del a` is executed.

### 深拷贝
`copy` 方法会完全复制数组及其数据。

```python
>>> d = a.copy()  # 创建一个具有新数据的新数组对象
>>> d is a
False
>>> d.base is a  # d 不与 a 共享任何东西
False
>>> d[0, 0] = 9999
>>> a
array([[   0,   10,   10,    3],
       [1234,   10,   10,    7],
       [   8,   10,   10,   11]])
```

如果原始数组不再需要，有时在切片后应该调用 `copy`。例如，假设 `a` 是一个巨大的中间结果，而最终结果 `b` 只包含 `a` 的一小部分，那么在通过切片构造 `b` 时应该进行深拷贝：

```python
>>> a = np.arange(int(1e8))
>>> b = a[:100].copy()
>>> del a  # 可以释放 ``a`` 的内存。
```

如果使用 `b = a[:100]` 而不是上述代码，`b` 将引用 `a`，即使执行了 `del a`，`a` 也会继续存在于内存中。

### Functions and methods overview
Here is a list of some useful NumPy functions and methods names ordered in categories. See [Routines and objects by topic](https://numpy.org/devdocs/reference/routines.html#routines) for the full list.

**Array Creation**
`arange`, `array`, `copy`, `empty`, `empty_like`, `eye`, `fromfile`, `fromfunction`, `identity`, `linspace`, `logspace`, `mgrid`, `ogrid`, `ones`, `ones_like`, `r_`, `zeros`, `zeros_like`

**Conversions**
`ndarray.astype`, `atleast_1d`, `atleast_2d`, `atleast_3d`, `mat`

**Manipulations**
`array_split`, `column_stack`, `concatenate`, `diagonal`, `dsplit`, `dstack`, `hsplit`, `hstack`, `ndarray.item`, `newaxis`, `ravel`, `repeat`, `reshape`, `resize`, `squeeze`, `swapaxes`, `take`, `transpose`, `vsplit`, `vstack`

**Questions**
`all`, `any`, `nonzero`, `where`

**Ordering**
`argmax`, `argmin`, `argsort`, `max`, `min`, `ptp`, `searchsorted`, `sort`

**Operations**
`choose`, `compress`, `cumprod`, `cumsum`, `inner`, `ndarray.fill`, `imag`, `prod`, `put`, `putmask`, `real`, `sum`

**Basic Statistics**
`cov`, `mean`, `std`, `var`

**Basic Linear Algebra**
`cross`, `dot`, `outer`, `linalg.svd`, `vdot`

### 函数和方法概述
以下是一些按类别排序的NumPy函数和方法名称列表。要查看完整列表，请访问[按主题分类的例程和对象](https://numpy.org/devdocs/reference/routines.html#routines)。

**数组创建**
`arange`、`array`、`copy`、`empty`、`empty_like`、`eye`、`fromfile`、`fromfunction`、`identity`、`linspace`、`logspace`、`mgrid`、`ogrid`、`ones`、`ones_like`、`r_`、`zeros`、`zeros_like`

**转换**
`ndarray.astype`、`atleast_1d`、`atleast_2d`、`atleast_3d`、`mat`

**操作**
`array_split`、`column_stack`、`concatenate`、`diagonal`、`dsplit`、`dstack`、`hsplit`、`hstack`、`ndarray.item`、`newaxis`、`ravel`、`repeat`、`reshape`、`resize`、`squeeze`、`swapaxes`、`take`、`transpose`、`vsplit`、`vstack`

**查询**
`all`、`any`、`nonzero`、`where`

**排序**
`argmax`、`argmin`、`argsort`、`max`、`min`、`ptp`、`searchsorted`、`sort`

**运算**
`choose`、`compress`、`cumprod`、`cumsum`、`inner`、`ndarray.fill`、`imag`、`prod`、`put`、`putmask`、`real`、`sum`

**基本统计**
`cov`、`mean`、`std`、`var`

**基本线性代数**
`cross`、`dot`、`outer`、`linalg.svd`、`vdot`

## Less basic

### Broadcasting rules
Broadcasting allows universal functions to deal in a meaningful way with inputs that do not have exactly the same shape.

The first rule of broadcasting is that if all input arrays do not have the same number of dimensions, a “1” will be repeatedly prepended to the shapes of the smaller arrays until all the arrays have the same number of dimensions.

The second rule of broadcasting ensures that arrays with a size of 1 along a particular dimension act as if they had the size of the array with the largest shape along that dimension. The value of the array element is assumed to be the same along that dimension for the “broadcast” array.

After application of the broadcasting rules, the sizes of all arrays must match. More details can be found in [Broadcasting](https://numpy.org/devdocs/user/basics.broadcasting.html#basics-broadcasting).

## 更高级的内容

### 广播规则
广播允许通用函数以有意义的方式处理形状不完全相同的输入。

广播的第一条规则是，如果所有输入数组不具有相同数量的维度，则会在较小数组的形状前反复添加“1”，直到所有数组都具有相同数量的维度。

广播的第二条规则确保了在某一特定维度上大小为1的数组，其行为就好像它们在该维度上具有最大形状数组的大小一样。对于“广播”数组，假定该维度上数组元素的值是相同的。

在应用广播规则后，所有数组的大小必须匹配。更多详细信息可以在[广播](https://numpy.org/devdocs/user/basics.broadcasting.html#basics-broadcasting)中找到。

## Advanced indexing and index tricks
NumPy offers more indexing facilities than regular Python sequences. In addition to indexing by integers and slices, as we saw before, arrays can be indexed by arrays of integers and arrays of booleans.

## 高级索引和索引技巧
NumPy 提供了比常规 Python 序列更多的索引功能。除了我们之前看到的通过整数和切片进行索引外，数组还可以通过整数数组和布尔数组进行索引。

### Indexing with arrays of indices

```python
>>> a = np.arange(12)**2  # the first 12 square numbers
>>> i = np.array([1, 1, 3, 8, 5])  # an array of indices
>>> a[i]  # the elements of `a` at the positions `i`
array([ 1,  1,  9, 64, 25])
>>> 
>>> j = np.array([[3, 4], [9, 7]])  # a bidimensional array of indices
>>> a[j]  # the same shape as `j`
array([[ 9, 16],
       [81, 49]])
```

When the indexed array `a` is multidimensional, a single array of indices refers to the first dimension of `a`. The following example shows this behavior by converting an image of labels into a color image using a palette.

```python
>>> palette = np.array([[0, 0, 0],         # black
...                     [255, 0, 0],       # red
...                     [0, 255, 0],       # green
...                     [0, 0, 255],       # blue
...                     [255, 255, 255]])  # white
>>> image = np.array([[0, 1, 2, 0],  # each value corresponds to a color in the palette
...                   [0, 3, 4, 0]])
>>> palette[image]  # the (2, 4, 3) color image
array([[[  0,   0,   0],
        [255,   0,   0],
        [  0, 255,   0],
        [  0,   0,   0]],

       [[  0,   0,   0],
        [  0,   0, 255],
        [255, 255, 255],
        [  0,   0,   0]]])
```

We can also give indexes for more than one dimension. The arrays of indices for each dimension must have the same shape.

```python
>>> a = np.arange(12).reshape(3, 4)
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> i = np.array([[0, 1],  # indices for the first dim of `a`
...               [1, 2]])
>>> j = np.array([[2, 1],  # indices for the second dim
...               [3, 3]])
>>> 
>>> a[i, j]  # i and j must have equal shape
array([[ 2,  5],
       [ 7, 11]])
>>> 
>>> a[i, 2]
array([[ 2,  6],
       [ 6, 10]])
>>> 
>>> a[:, j]
array([[[ 2,  1],
        [ 3,  3]],

       [[ 6,  5],
        [ 7,  7]],

       [[10,  9],
        [11, 11]]])
```

In Python, `arr[i, j]` is exactly the same as `arr[(i, j)]`—so we can put `i` and `j` in a `tuple` and then do the indexing with that.

```python
>>> l = (i, j)
>>> # equivalent to a[i, j]
>>> a[l]
array([[ 2,  5],
       [ 7, 11]])
```

However, we can not do this by putting `i` and `j` into an array, because this array will be interpreted as indexing the first dimension of `a`.

```python
>>> s = np.array([i, j])
>>> # not what we want
>>> a[s]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: index 3 is out of bounds for axis 0 with size 3
>>> # same as `a[i, j]`
>>> a[tuple(s)]
array([[ 2,  5],
       [ 7, 11]])
```

Another common use of indexing with arrays is the search of the maximum value of time-dependent series:

```python
>>> time = np.linspace(20, 145, 5)  # time scale
>>> data = np.sin(np.arange(20)).reshape(5, 4)  # 4 time-dependent series
>>> time
array([ 20.  ,  51.25,  82.5 , 113.75, 145.  ])
>>> data
array([[ 0.        ,  0.84147098,  0.90929743,  0.14112001],
       [-0.7568025 , -0.95892427, -0.2794155 ,  0.6569866 ],
       [ 0.98935825,  0.41211849, -0.54402111, -0.99999021],
       [-0.53657292,  0.42016704,  0.99060736,  0.65028784],
       [-0.28790332, -0.96139749, -0.75098725,  0.14987721]])
>>> # index of the maxima for each series
>>> ind = data.argmax(axis=0)
>>> ind
array([2, 0, 3, 1])
>>> # times corresponding to the maxima
>>> time_max = time[ind]
>>> 
>>> data_max = data[ind, range(data.shape[1])]  # => data[ind[0], 0], data[ind[1], 1]...
>>> time_max
array([ 82.5 ,  20.  , 113.75,  51.25])
>>> data_max
array([0.98935825, 0.84147098, 0.99060736, 0.6569866 ])
>>> np.all(data_max == data.max(axis=0))
True
```

You can also use indexing with arrays as a target to assign to:

```python
>>> a = np.arange(5)
>>> a
array([0, 1, 2, 3, 4])
>>> a[[1, 3, 4]] = 0
>>> a
array([0, 0, 2, 0, 0])
```

However, when the list of indices contains repetitions, the assignment is done several times, leaving behind the last value:

```python
>>> a = np.arange(5)
>>> a[[0, 0, 2]] = [1, 2, 3]
>>> a
array([2, 1, 3, 3, 4])
```

This is reasonable enough, but watch out if you want to use Python’s += construct, as it may not do what you expect:

```python
>>> a = np.arange(5)
>>> a[[0, 0, 2]] += 1
>>> a
array([1, 1, 3, 3, 4])
```

Even though 0 occurs twice in the list of indices, the 0th element is only incremented once. This is because Python requires `a += 1` to be equivalent to `a = a + 1`.

### 使用索引数组进行索引

```python
>>> a = np.arange(12)**2  # 前12个平方数
>>> i = np.array([1, 1, 3, 8, 5])  # 一个索引数组
>>> a[i]  # `a` 中位置为 `i` 的元素
array([ 1,  1,  9, 64, 25])
>>> 
>>> j = np.array([[3, 4], [9, 7]])  # 一个二维索引数组
>>> a[j]  # 与 `j` 形状相同的数组
array([[ 9, 16],
       [81, 49]])
```

当被索引的数组 `a` 是多维的，一个单独的索引数组会引用 `a` 的第一维。下面的例子通过使用调色板将一个标签图像转换为彩色图像来展示了这种行为。

```python
>>> palette = np.array([[0, 0, 0],         # 黑色
...                     [255, 0, 0],       # 红色
...                     [0, 255, 0],       # 绿色
...                     [0, 0, 255],       # 蓝色
...                     [255, 255, 255]])  # 白色
>>> image = np.array([[0, 1, 2, 0],  # 每个值对应于调色板中的颜色
...                   [0, 3, 4, 0]])
>>> palette[image]  # (2, 4, 3) 的彩色图像
array([[[  0,   0,   0],
        [255,   0,   0],
        [  0, 255,   0],
        [  0,   0,   0]],

       [[  0,   0,   0],
        [  0,   0, 255],
        [255, 255, 255],
        [  0,   0,   0]]])
```

我们还可以为多个维度提供索引。每个维度的索引数组必须具有相同的形状。

```python
>>> a = np.arange(12).reshape(3, 4)
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> i = np.array([[0, 1],  # `a` 第一维的索引
...               [1, 2]])
>>> j = np.array([[2, 1],  # `a` 第二维的索引
...               [3, 3]])
>>> 
>>> a[i, j]  # i 和 j 必须具有相同的形状
array([[ 2,  5],
       [ 7, 11]])
>>> 
>>> a[i, 2]  # 第二维索引被替换为整数 2
array([[ 2,  6],
       [ 6, 10]])
>>> 
>>> a[:, j]  # 选择了所有行的第 j 列
array([[[ 2,  1],
        [ 3,  3]],

       [[ 6,  5],
        [ 7,  7]],

       [[10,  9],
        [11, 11]]])
```

在 Python 中，`arr[i, j]` 与 `arr[(i, j)]` 是完全相同的——因此我们可以将 `i` 和 `j` 放入一个 `tuple` 中，然后用这个 `tuple` 进行索引。

```python
>>> l = (i, j)
>>> # 等同于 a[i, j]
>>> a[l]
array([[ 2,  5],
       [ 7, 11]])
```

然而，我们不能简单地将 `i` 和 `j` 放入一个数组中并尝试这样做，因为这个数组会被解释为索引 `a` 的第一维。

```python
>>> s = np.array([i, j])
>>> # 这不是我们想要的
>>> a[s]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: index 3 is out of bounds for axis 0 with size 3
>>> # 等同于 `a[i, j]`
>>> a[tuple(s)]  # 注意这里会出错，因为 s 包含了两个数组而不是一个索引元组
array([[ 2,  5],
       [ 7, 11]])
```

使用数组索引的另一种常见用途是搜索依赖于时间序列的最大值：

```python
>>> time = np.linspace(20, 145, 5)  # 时间尺度
>>> data = np.sin(np.arange(20)).reshape(5, 4)  # 4个依赖于时间的序列
>>> time
array([ 20.  ,  51.25,  82.5 , 113.75, 145.  ])
>>> data
array([[ 0.        ,  0.84147098,  0.90929743,  0.14112001],
       [-0.7568025 , -0.95892427, -0.2794155 ,  0.6569866 ],
       [ 0.98935825,  0.41211849, -0.54402111, -0.99999021],
       [-0.53657292,  0.42016704,  0.99060736,  0.65028784],
       [-0.28790332, -0.96139749, -0.75098725,  0.14987721]])
>>> # 每个序列的最大值的索引
>>> ind = data.argmax(axis=0)
>>> ind
array([2, 0, 3, 1])
>>> # 与最大值对应的时间
>>> time_max = time[ind]
>>> 
>>> data_max = data[ind, range(data.shape[1])]  # => data[ind[0], 0], data[ind[1], 1]...
>>> time_max
array([ 82.5 ,  20.  , 113.75,  51.25])
>>> data_max
array([0.98935825, 0.84147098, 0.99060736, 0.6569866 ])
>>> np.all(data_max == data.max(axis=0))
True
```

你还可以使用数组索引作为赋值的目标：

```python
>>> a = np.arange(5)
>>> a
array([0, 1, 2, 3, 4])
>>> a[[1, 3, 4]] = 0
>>> a
array([0, 0, 2, 0, 0])
```

然而，当索引列表中包含重复时，赋值会进行多次，最终保留最后一个值：

```python
>>> a = np.arange(5)
>>> a[[0, 0, 2]] = [1, 2, 3]
>>> a
array([2, 1, 3, 3, 4])
```

这是相当合理的，但如果你想要使用Python的`+=`结构，请注意它可能不会按你预期的方式工作：

```python
>>> a = np.arange(5)
>>> a[[0, 0, 2]] += 1
>>> a
array([1, 1, 3, 3, 4])
```

尽管索引列表中0出现了两次，但第0个元素只增加了一次。这是因为Python要求`a += 1`等价于`a = a + 1`。

### Indexing with boolean arrays
When we index arrays with arrays of (integer) indices we are providing the list of indices to pick. With boolean indices the approach is different; we explicitly choose which items in the array we want and which ones we don’t.

The most natural way one can think of for boolean indexing is to use boolean arrays that have the same shape as the original array:

```python
>>> a = np.arange(12).reshape(3, 4)
>>> b = a > 4
>>> b  # `b` is a boolean with `a`'s shape
array([[False, False, False, False],
       [False,  True,  True,  True],
       [ True,  True,  True,  True]])
>>> a[b]  # 1d array with the selected elements
array([ 5,  6,  7,  8,  9, 10, 11])
```

This property can be very useful in assignments:

```python
>>> a[b] = 0  # All elements of `a` higher than 4 become 0
>>> a
array([[0, 1, 2, 3],
       [4, 0, 0, 0],
       [0, 0, 0, 0]])
```

You can look at the following example to see how to use boolean indexing to generate an image of the [Mandelbrot set](https://en.wikipedia.org/wiki/Mandelbrot_set):

```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> def mandelbrot(h, w, maxit=20, r=2):
    """Returns an image of the Mandelbrot fractal of size (h,w)."""
    x = np.linspace(-2.5, 1.5, 4*h+1)
    y = np.linspace(-1.5, 1.5, 3*w+1)
    A, B = np.meshgrid(x, y)
    C = A + B*1j
    z = np.zeros_like(C)
    divtime = maxit + np.zeros(z.shape, dtype=int)

    for i in range(maxit):
        z = z**2 + C
        diverge = abs(z) > r                    # who is diverging
        div_now = diverge & (divtime == maxit)  # who is diverging now
        divtime[div_now] = i                    # note when
        z[diverge] = r                          # avoid diverging too much

    return divtime
>>> plt.clf()
>>> plt.imshow(mandelbrot(400, 400))
```

![1](https://numpy.org/devdocs/_images/quickstart-1.png)

The second way of indexing with booleans is more similar to integer indexing; for each dimension of the array we give a 1D boolean array selecting the slices we want:

```python
>>> a = np.arange(12).reshape(3, 4)
>>> b1 = np.array([False, True, True])         # first dim selection
>>> b2 = np.array([True, False, True, False])  # second dim selection
>>> 
>>> a[b1, :]                                   # selecting rows
array([[ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> 
>>> a[b1]                                      # same thing
array([[ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> 
>>> a[:, b2]                                   # selecting columns
array([[ 0,  2],
       [ 4,  6],
       [ 8, 10]])
>>> 
>>> a[b1, b2]                                  # a weird thing to do
array([ 4, 10])
```

Note that the length of the 1D boolean array must coincide with the length of the dimension (or axis) you want to slice. In the previous example, `b1` has length 3 (the number of rows in `a`), and `b2` (of length 4) is suitable to index the 2nd axis (columns) of `a`.

### 使用布尔数组索引
当我们使用（整数）索引数组来索引数组时，我们提供了要选择的索引列表。然而，使用布尔索引时，方法就不同了；我们明确地选择数组中的哪些项是我们想要的，哪些是不想要的。

对于布尔索引，最自然的方法是使用与原始数组形状相同的布尔数组：

```python
>>> a = np.arange(12).reshape(3, 4)
>>> b = a > 4
>>> b  # `b` 是一个与 `a` 形状相同的布尔数组
array([[False, False, False, False],
       [False,  True,  True,  True],
       [ True,  True,  True,  True]])
>>> a[b]  # 包含选定元素的一维数组
array([ 5,  6,  7,  8,  9, 10, 11])
```

这个特性在赋值时非常有用：

```python
>>> a[b] = 0  # `a` 中大于 4 的所有元素变为 0
>>> a
array([[0, 1, 2, 3],
       [4, 0, 0, 0],
       [0, 0, 0, 0]])
```

你可以查看下面的例子来了解如何使用布尔索引生成[Mandelbrot 集](https://en.wikipedia.org/wiki/Mandelbrot_set)的图像：

```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> def mandelbrot(h, w, maxit=20, r=2):
    """返回大小为 (h,w) 的 Mandelbrot 分形图像。"""
    x = np.linspace(-2.5, 1.5, 4*h+1)
    y = np.linspace(-1.5, 1.5, 3*w+1)
    A, B = np.meshgrid(x, y)
    C = A + B*1j
    z = np.zeros_like(C)
    divtime = maxit + np.zeros(z.shape, dtype=int)

    for i in range(maxit):
        z = z**2 + C
        diverge = abs(z) > r                    # 哪些在发散
        div_now = diverge & (divtime == maxit)  # 现在哪些在发散
        divtime[div_now] = i                    # 记录何时发散
        z[diverge] = r                          # 避免过度发散

    return divtime
>>> plt.clf()
>>> plt.imshow(mandelbrot(400, 400))
```

![1](https://numpy.org/devdocs/_images/quickstart-1.png)

使用布尔值进行索引的第二种方式与整数索引更为相似；我们为数组的每一维提供一个一维布尔数组，以选择我们想要的切片：

```python
>>> a = np.arange(12).reshape(3, 4)
>>> b1 = np.array([False, True, True])         # 第一维的选择
>>> b2 = np.array([True, False, True, False])  # 第二维的选择
>>> 
>>> a[b1, :]                                   # 选择行
array([[ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> 
>>> a[b1]                                      # 效果相同
array([[ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> 
>>> a[:, b2]                                   # 选择列
array([[ 0,  2],
       [ 4,  6],
       [ 8, 10]])
>>> 
>>> a[b1, b2]                                  # 一个奇怪的操作
array([ 4, 10])
```

请注意，一维布尔数组的长度必须与你要切片的维度（或轴）的长度相匹配。在前面的例子中，`b1` 的长度为 3（`a` 的行数），而 `b2`（长度为 4）适合索引 `a` 的第二轴（列）。

### The ix_() function
The `ix_` function can be used to combine different vectors so as to obtain the result for each n-uplet. For example, if you want to compute all the a+b*c for all the triplets taken from each of the vectors a, b and c:

```python
>>> a = np.array([2, 3, 4, 5])
>>> b = np.array([8, 5, 4])
>>> c = np.array([5, 4, 6, 8, 3])
>>> ax, bx, cx = np.ix_(a, b, c)
>>> ax
array([[[2]],

       [[3]],

       [[4]],

       [[5]]])
>>> bx
array([[[8],
        [5],
        [4]]])
>>> cx
array([[[5, 4, 6, 8, 3]]])
>>> ax.shape, bx.shape, cx.shape
((4, 1, 1), (1, 3, 1), (1, 1, 5))
>>> result = ax + bx * cx
>>> result
array([[[42, 34, 50, 66, 26],
        [27, 22, 32, 42, 17],
        [22, 18, 26, 34, 14]],

       [[43, 35, 51, 67, 27],
        [28, 23, 33, 43, 18],
        [23, 19, 27, 35, 15]],

       [[44, 36, 52, 68, 28],
        [29, 24, 34, 44, 19],
        [24, 20, 28, 36, 16]],

       [[45, 37, 53, 69, 29],
        [30, 25, 35, 45, 20],
        [25, 21, 29, 37, 17]]])
>>> result[3, 2, 4]
17
>>> a[3] + b[2] * c[4]
17
```

You could also implement the reduce as follows:

```python
>>> def ufunc_reduce(ufct, *vectors):
...    vs = np.ix_(*vectors)
...    r = ufct.identity
...    for v in vs:
...        r = ufct(r, v)
...    return r
```

and then use it as:

```python
>>> ufunc_reduce(np.add, a, b, c)
array([[[15, 14, 16, 18, 13],
        [12, 11, 13, 15, 10],
        [11, 10, 12, 14,  9]],

       [[16, 15, 17, 19, 14],
        [13, 12, 14, 16, 11],
        [12, 11, 13, 15, 10]],

       [[17, 16, 18, 20, 15],
        [14, 13, 15, 17, 12],
        [13, 12, 14, 16, 11]],

       [[18, 17, 19, 21, 16],
        [15, 14, 16, 18, 13],
        [14, 13, 15, 17, 12]]])
```

The advantage of this version of reduce compared to the normal ufunc.reduce is that it makes use of the [broadcasting rules](https://numpy.org/devdocs/user/quickstart.html#broadcasting-rules) in order to avoid creating an argument array the size of the output times the number of vectors.

### `ix_` 函数
`ix_` 函数可以用于组合不同的向量，以便为每个 n 元组获取结果。例如，如果你想为从向量 a、b 和 c 中取出的每个三元组计算所有 a+b*c 的值：

```python
>>> a = np.array([2, 3, 4, 5])
>>> b = np.array([8, 5, 4])
>>> c = np.array([5, 4, 6, 8, 3])
>>> ax, bx, cx = np.ix_(a, b, c)
>>> ax
array([[[2]],

       [[3]],

       [[4]],

       [[5]]])
>>> bx
array([[[8],
        [5],
        [4]]])
>>> cx
array([[[5, 4, 6, 8, 3]]])
>>> ax.shape, bx.shape, cx.shape
((4, 1, 1), (1, 3, 1), (1, 1, 5))
>>> result = ax + bx * cx
>>> result
array([[[42, 34, 50, 66, 26],
        [27, 22, 32, 42, 17],
        [22, 18, 26, 34, 14]],

       [[43, 35, 51, 67, 27],
        [28, 23, 33, 43, 18],
        [23, 19, 27, 35, 15]],

       [[44, 36, 52, 68, 28],
        [29, 24, 34, 44, 19],
        [24, 20, 28, 36, 16]],

       [[45, 37, 53, 69, 29],
        [30, 25, 35, 45, 20],
        [25, 21, 29, 37, 17]]])
>>> result[3, 2, 4]
17
>>> a[3] + b[2] * c[4]
17
```

你也可以这样实现 reduce 函数：

```python
>>> def ufunc_reduce(ufct, *vectors):
...    vs = np.ix_(*vectors)
...    r = ufct.identity  # 获取ufunc的恒等元素
...    for v in vs:
...        r = ufct(r, v)  # 对每个扩展后的向量进行累积运算
...    return r
```

然后你可以像这样使用它：

```python
>>> ufunc_reduce(np.add, a, b, c)
array([[[15, 14, 16, 18, 13],
        [12, 11, 13, 15, 10],
        [11, 10, 12, 14,  9]],

       [[16, 15, 17, 19, 14],
        [13, 12, 14, 16, 11],
        [12, 11, 13, 15, 10]],

       [[17, 16, 18, 20, 15],
        [14, 13, 15, 17, 12],
        [13, 12, 14, 16, 11]],

       [[18, 17, 19, 21, 16],
        [15, 14, 16, 18, 13],
        [14, 13, 15, 17, 12]]])
```

与普通的 ufunc.reduce 相比，这个版本的 reduce 的优势在于它利用了 [广播规则](https://numpy.org/devdocs/user/quickstart.html#broadcasting-rules)，从而避免了创建一个大小为输出乘以向量数量的参数数组。这样可以显著减少内存使用和提高计算效率。

### Indexing with strings
See [Structured arrays](https://numpy.org/devdocs/user/basics.rec.html#structured-arrays).

### 使用字符串索引
请参阅[结构化数组](https://numpy.org/devdocs/user/basics.rec.html#structured-arrays)。

## Tricks and tips
Here we give a list of short and useful tips.

## 技巧和提示
这里我们列出了一些简短且有用的提示。

### “Automatic” reshaping
To change the dimensions of an array, you can omit one of the sizes which will then be deduced automatically:

```python
>>> a = np.arange(30)
>>> b = a.reshape((2, -1, 3))  # -1 means "whatever is needed"
>>> b.shape
(2, 5, 3)
>>> b
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11],
        [12, 13, 14]],

       [[15, 16, 17],
        [18, 19, 20],
        [21, 22, 23],
        [24, 25, 26],
        [27, 28, 29]]])
```

### “自动”重塑
要更改数组的维度，可以省略其中一个大小，它会被自动推断：

```python
>>> a = np.arange(30)
>>> b = a.reshape((2, -1, 3))  # -1 表示“需要的任意值”
>>> b.shape
(2, 5, 3)
>>> b
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11],
        [12, 13, 14]],

       [[15, 16, 17],
        [18, 19, 20],
        [21, 22, 23],
        [24, 25, 26],
        [27, 28, 29]]])
```

### Vector stacking
How do we construct a 2D array from a list of equally-sized row vectors? In MATLAB this is quite easy: if `x` and `y` are two vectors of the same length you only need do `m=[x;y]`. In NumPy this works via the functions `column_stack`, `dstack`, `hstack` and `vstack`, depending on the dimension in which the stacking is to be done. For example:

```python
>>> x = np.arange(0, 10, 2)
>>> y = np.arange(5)
>>> m = np.vstack([x, y])
>>> m
array([[0, 2, 4, 6, 8],
       [0, 1, 2, 3, 4]])
>>> xy = np.hstack([x, y])
>>> xy
array([0, 2, 4, 6, 8, 0, 1, 2, 3, 4])
```

The logic behind those functions in more than two dimensions can be strange.

### 向量堆叠
我们如何从一系列大小相同的行向量中构造一个2D数组？在MATLAB中，这很简单：如果`x`和`y`是两个长度相同的向量，你只需要做`m=[x;y]`。在NumPy中，这通过`column_stack`、`dstack`、`hstack`和`vstack`函数实现，具体取决于堆叠的维度。例如：

```python
>>> x = np.arange(0, 10, 2)
>>> y = np.arange(5)
>>> m = np.vstack([x, y])
>>> m
array([[0, 2, 4, 6, 8],
       [0, 1, 2, 3, 4]])
>>> xy = np.hstack([x, y])
>>> xy
array([0, 2, 4, 6, 8, 0, 1, 2, 3, 4])
```

在超过两个维度时，这些函数的逻辑可能会有些奇怪。

### Histograms
The NumPy `histogram` function applied to an array returns a pair of vectors: the histogram of the array and a vector of the bin edges. Beware: `matplotlib` also has a function to build histograms (called `hist`, as in Matlab) that differs from the one in NumPy. The main difference is that `pylab.hist` plots the histogram automatically, while `numpy.histogram` only generates the data.

```python
>>> import numpy as np
>>> rg = np.random.default_rng(1)
>>> import matplotlib.pyplot as plt
>>> # Build a vector of 10000 normal deviates with variance 0.5^2 and mean 2
>>> mu, sigma = 2, 0.5
>>> v = rg.normal(mu, sigma, 10000)
>>> # Plot a normalized histogram with 50 bins
>>> plt.hist(v, bins=50, density=True)       # matplotlib version (plot)
(array...)
>>> # Compute the histogram with numpy and then plot it
>>> (n, bins) = np.histogram(v, bins=50, density=True)  # NumPy version (no plot)
>>> plt.plot(.5 * (bins[1:] + bins[:-1]), n) 
```

![2](https://numpy.org/devdocs/_images/quickstart-2.png)

With Matplotlib >=3.4 you can also use `plt.stairs(n, bins)`.

### 直方图
NumPy 的 `histogram` 函数应用于数组时，会返回一对向量：数组的直方图和箱子的边缘向量。请注意：`matplotlib` 也有一个用于构建直方图的函数（称为 `hist`，类似于 Matlab 中的同名函数），但它与 NumPy 中的函数有所不同。主要区别在于 `pylab.hist` 会自动绘制直方图，而 `numpy.histogram` 仅生成数据。

```python
>>> import numpy as np
>>> rg = np.random.default_rng(1)
>>> import matplotlib.pyplot as plt
>>> # 创建一个包含 10000 个正态分布变量的向量，方差为 0.5^2，均值为 2
>>> mu, sigma = 2, 0.5
>>> v = rg.normal(mu, sigma, 10000)
>>> # 绘制一个有 50 个箱子的归一化直方图
>>> plt.hist(v, bins=50, density=True)       # matplotlib 版本（绘图）
(array...)
>>> # 使用 numpy 计算直方图，然后绘制它
>>> (n, bins) = np.histogram(v, bins=50, density=True)  # NumPy 版本（不绘图）
>>> plt.plot(.5 * (bins[1:] + bins[:-1]), n) 
```

![2](https://numpy.org/devdocs/_images/quickstart-2.png)

在 Matplotlib >=3.4 中，你还可以使用 `plt.stairs(n, bins)` 来绘制直方图。