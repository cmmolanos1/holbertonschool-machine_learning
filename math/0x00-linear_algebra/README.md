# Linear Algebra

Learning the fundamentals of this type of algebra, its operations, vectors and matrices.

## Tasks
### 0. Slice Me Up
File: 0-slice_me_up.py

Printing the first two numbers, the last five numbers, and the interval between the 2th and 6th number of a List.

### 1. Trim Me Down
File: 1-trim_me_down.py

Prints the third and fourth column of a given matrix.

### 2. Size Me Please

File: 2-size_me_please.py

Function that calculates the shape(dimension) of a given matrix.

Example:

```python
>>> print(matrix_shape([[1, 2], [3, 4]]))
[2, 2]
```

### 3. Flip Me Over

File: 3-flip_me_over.py

Function that returns the transpose of a given matrix.

Example:
```python
>>> print(matrix_transpose([[1, 2], [3, 4]]))
[[1, 3], [2, 4]]
```

### 4. Line Up

File: 4-line_up.py

Function that adds two arrays element-wise.

Example:

```python
>>> arr1 = [1, 2, 3, 4]
>>> arr2 = [5, 6, 7, 8]
>>> print(add_arrays(arr1, arr2))
[6, 8, 10, 12]
```

### 5. Across The Planes

File: 5-across_the_planes.py

Function that adds two 2D-matrices element-wise.

Example:

```python
>>> mat1 = [[1, 2], [3, 4]]
>>> mat2 = [[5, 6], [7, 8]]
>>> print(add_matrices2D(mat1, mat2))
[[6, 8], [10, 12]]
```

### 6. Howdy Partner 

File: 6-howdy_partner.py

Function that concatenates two arrays:

Example:

```python
>>> arr1 = [1, 2, 3, 4, 5]
>>> arr2 = [6, 7, 8]
>>> print(cat_arrays(arr1, arr2))
[1, 2, 3, 4, 5, 6, 7, 8]
```

### 7. Gettin’ Cozy

File: 7-gettin_cozy.py

Function that concatenates two matrices along a specific axis.

Example:

```python
>>> mat1 = [[1, 2], [3, 4]]
>>> mat2 = [[5, 6]]
>>> mat3 = [[7], [8]]
>>> mat4 = cat_matrices2D(mat1, mat2)
>>> mat5 = cat_matrices2D(mat1, mat3, axis=1)
>>> print(mat4)
[[1, 2], [3, 4], [5, 6]]
>>> print(mat5)
[[1, 2, 7], [3, 4, 8]]
```

### 8. Ridin’ Bareback

File: 8-ridin_bareback.py

Function that performs matrix multiplication.

Example:

```python
>>> mat1 = [[1, 2],
...         [3, 4],
...         [5, 6]]
>>> mat2 = [[1, 2, 3, 4],
...         [5, 6, 7, 8]]
>>> print(mat_mul(mat1, mat2))
[[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]
```

### 9. Let The Butcher Slice It 

File: 9-let_the_butcher_slice_it.py

Practice of numpy slicing.

### 10. I’ll Use My Scale 

File: 10-ill_use_my_scale.py

Function that calculates the shape of a numpy.ndarray.

Example:

```python
>>> import numpy as np
>>> mat1 = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
...                  [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]])
>>> print(np_shape(mat1))
(2, 2, 5)
```

### 11. The Western Exchange 

File: 11-the_western_exchange.py

Function that transposes n-dimensional matrix.

Example:

```python
>>> import numpy as np
>>> mat1 = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
...                  [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]])
>>> print(np_transpose(mat1))
[[[ 1 11]
  [ 6 16]]

 [[ 2 12]
  [ 7 17]]

 [[ 3 13]
  [ 8 18]]

 [[ 4 14]
  [ 9 19]]

 [[ 5 15]
  [10 20]]]
```

### 12. Bracing The Elements

File: 12-bracin_the_elements.py

Function that performs element-wise addition, subtraction, multiplication, and division.

Example:

```python
>>> import numpy as np
>>> mat1 = np.array([[11, 22, 33], [44, 55, 66]])
>>> mat2 = np.array([[1, 2, 3], [4, 5, 6]])
>>> add, sub, mul, div = np_elementwise(mat1, mat2)
>>> print("Add:\n", add, "\nSub:\n", sub, "\nMul:\n", mul, "\nDiv:\n", div)
Add:
 [[12 24 36]
 [48 60 72]] 
Sub:
 [[10 20 30]
 [40 50 60]] 
Mul:
 [[ 11  44  99]
 [176 275 396]] 
Div:
 [[11. 11. 11.]
 [11. 11. 11.]]
```

### 13. Cat's Got Your Tongue

File: 13-cats_got_your_tongue.py

Function that concatenates two matrices along a specific axis:.

Example:

```python
>>> import numpy as np
>>> mat1 = np.array([[11, 22, 33], [44, 55, 66]])
>>> mat2 = np.array([[1, 2, 3], [4, 5, 6]])
>>> print(np_cat(mat1, mat2))
[[11 22 33]
 [44 55 66]
 [ 1  2  3]
 [ 4  5  6]]
>>> print(np_cat(mat1, mat2, axis=1))
[[11 22 33  1  2  3]
 [44 55 66  4  5  6]]
```

### 14. Saddle Up

File: 14-saddle_up.py

Function that performs matrix multiplication.

Example:

```python
>>> import numpy as np
>>> mat1 = np.array([[11, 22, 33], [44, 55, 66]])
>>> mat2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
>>> print(np_matmul(mat1, mat2))
[[ 330  396  462]
 [ 726  891 1056]]
```

### Slice Like A Ninja

File: 

Function that slices a matrix along a specific axes.

Example:

```python
>>> import numpy as np
>>> mat2 = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
...                  [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
...                  [[21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]])
>>> print(np_slice(mat2, axes={0: (2,), 2: (None, None, -2)}))
[[ 1  2  3  4  5]
 [ 6  7  8  9 10]]
[[[ 5  3  1]
  [10  8  6]]

 [[15 13 11]
  [20 18 16]]
```

### 16. The Whole Barn

File: 101-the_whole_barn.py

Function that adds two n-dimensional matrices.

Example:

```python
>>> mat1 = [[1, 2], [3, 4]]
>>> mat2 = [[5, 6], [7, 8]]
>>> print(add_matrices(mat1, mat2))
[[6, 8], [10, 12]]
```
