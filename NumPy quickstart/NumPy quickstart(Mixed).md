# NumPy快速入门

## Prerequisites
You’ll need to know a bit of Python. For a refresher, see the [Python tutorial.](https://docs.python.org/tutorial/)

To work the examples, you’ll need `matplotlib` installed in addition to NumPy.

## 前提条件
你需要了解一些Python的基础知识。为了复习，请查阅[Python教程。](https://docs.python.org/tutorial/)

为了运行示例，除了NumPy外，你还需要安装`matplotlib`。

### Learner profile
This is a quick overview of arrays in NumPy. It demonstrates how n-dimensional (
) arrays are represented and can be manipulated. In particular, if you don’t know how to apply common functions to n-dimensional arrays (without using for-loops), or if you want to understand axis and shape properties for n-dimensional arrays, this article might be of help.

### 学习者画像
本文是对NumPy中数组的快速概述。它展示了n维数组是如何表示和操作的。特别是，如果你不知道如何对n维数组应用常见函数（不使用for循环），或者如果你想了解n维数组的轴和形状属性，本文可能会对你有所帮助。

### Learning Objectives
After reading, you should be able to:

* Understand the difference between one-, two- and n-dimensional arrays in NumPy;

* Understand how to apply some linear algebra operations to n-dimensional arrays without using for-loops;

* Understand axis and shape properties for n-dimensional arrays.

### 学习目标
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

## An example

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

## 一个示例

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

## Array creation

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

## 数组创建

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