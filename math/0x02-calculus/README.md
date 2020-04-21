# Calculus.

Learning:
 * Sigma(Σ) and pi(π) notation.
 * Derivative:
   -  Polynomics.
   - Exponents.
   - Logaritms.
   - Chain Rule.
   - Partial derivatives.
 * Integral:
   - Indefinite.
   - Definite.
   - Multiple Integral.
   - Double Integral.
   
## Tasks:

### 0. Sigma is for Sum

Solve the next summation:

![\sum_{i=2}^{5} i](https://render.githubusercontent.com/render/math?math=%5Csum_%7Bi%3D2%7D%5E%7B5%7D%20i)

### 1. It's actually pronounced sEEgma

Solve the next summation:

![\sum_{k=1}^{4} 9i - 2k](https://render.githubusercontent.com/render/math?math=%5Csum_%7Bk%3D1%7D%5E%7B4%7D%209i%20-%202k)   

### 2. Pi is for Product

Solve the next repeated multiplication:

![\prod_{i=1}^{m} i](https://render.githubusercontent.com/render/math?math=%5Cprod_%7Bi%3D1%7D%5E%7Bm%7D%20i)

### 3. It's actually pronounced pEE

Solve the next repeated multiplication:

![\prod_{i=0}^{10} i](https://render.githubusercontent.com/render/math?math=%5Cprod_%7Bi%3D0%7D%5E%7B10%7D%20i)

### 4. Hello, derivatives!

Found  ![\frac{dy}{dx}](https://render.githubusercontent.com/render/math?math=%5Cfrac%7Bdy%7D%7Bdx%7D), where:

![y = x^{4} + 3x^{3} - 5x + 1](https://render.githubusercontent.com/render/math?math=y%20%3D%20x%5E%7B4%7D%20%2B%203x%5E%7B3%7D%20-%205x%20%2B%201)

### 5. A log on the fire

Find derivative of:

![\frac{d(xln(x))}{dx}](https://render.githubusercontent.com/render/math?math=%5Cfrac%7Bd(xln(x))%7D%7Bdx%7D)

### 6. It is difficult to free fools from the chains they revere 

Find derivative of:

![\frac{d(ln(x^{2}))}{dx}](https://render.githubusercontent.com/render/math?math=%5Cfrac%7Bd(ln(x%5E%7B2%7D))%7D%7Bdx%7D)

### 7. Partial truths are often more insidious than total falsehoods 

Find  ![\frac{\partial}{\partial y} f(x,y)](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20y%7D%20f(x%2Cy)), where:

![f(x,y) = e^{xy}](https://render.githubusercontent.com/render/math?math=f(x%2Cy)%20%3D%20e%5E%7Bxy%7D)

and

![\frac{\partial x}{\partial y} = \frac{\partial y}{\partial x} = 0](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%20x%7D%7B%5Cpartial%20y%7D%20%3D%20%5Cfrac%7B%5Cpartial%20y%7D%7B%5Cpartial%20x%7D%20%3D%200) 

### 8. Put it all together and what do you get?

Find ![\frac{\partial}{\partial y \partial x} e^{x^{2}y}](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20y%20%5Cpartial%20x%7D%20e%5E%7Bx%5E%7B2%7Dy%7D), where:

![\frac{\partial x}{\partial y} = \frac{\partial y}{\partial x} = 0](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%20x%7D%7B%5Cpartial%20y%7D%20%3D%20%5Cfrac%7B%5Cpartial%20y%7D%7B%5Cpartial%20x%7D%20%3D%200) 

###  9. Our life is the sum total of all the decisions we make every day, and those decisions are determined by our priorities

* File:  9-sum_total.py

Function that calculates ![\sum_{i=1}^{n} i^{2}](https://render.githubusercontent.com/render/math?math=%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20i%5E%7B2%7D).

Example:

```python
>>> n = 5
>>> print(summation_i_squared(n))
55
```

### 10. Derive happiness in oneself from a good day's work

Function `def poly_derivative(poly):` that calculates the derivative of a polynomial. Where poly is a list of coefficients representing a polynomial.

Example:

f(x) = x³ + 3x + 5 

↓         ↓      ↓

f(x) = 5 + 3x  + x³ --->  f'(x) = 3 + 3x²

`poly = [5, 3, 0, 3]`

```python
>>> poly = [5, 3, 0, 1]
>>> print(poly_derivative(poly))
[3, 0, 3]
```

### 11. Good grooming is integral and impeccable style is a must 

Find the antiderivative of:

![\int x^{3}dx](https://render.githubusercontent.com/render/math?math=%5Cint%20x%5E%7B3%7Ddx)

### 12. We are all an integral part of the web of life

Find the antiderivative of:

![\int e^{2y}dy](https://render.githubusercontent.com/render/math?math=%5Cint%20e%5E%7B2y%7Ddy)

### 13. Create a definite plan for carrying out your desire and begin at once

Find the definite integration of:

![\int_{0}^{3}u^{2}du](https://render.githubusercontent.com/render/math?math=%5Cint_%7B0%7D%5E%7B3%7Du%5E%7B2%7Ddu)

### 14. My talents fall within definite limitations 

Find the definite integration of:

![\int_{-1}^{0}\frac{1}{v}dv](https://render.githubusercontent.com/render/math?math=%5Cint_%7B-1%7D%5E%7B0%7D%5Cfrac%7B1%7D%7Bv%7Ddv)

### 15. Winners are people with definite purpose in life

Find the definite integration of:

![\int_{0}^{5}xdy](https://render.githubusercontent.com/render/math?math=%5Cint_%7B0%7D%5E%7B5%7Dxdy)

### 16. Double whammy 

Answer the next double integration:

![\int_{1}^{2}\int_{0}^{3}x^{2}y^{-1}dxdy](https://render.githubusercontent.com/render/math?math=%5Cint_%7B1%7D%5E%7B2%7D%5Cint_%7B0%7D%5E%7B3%7Dx%5E%7B2%7Dy%5E%7B-1%7Ddxdy)

### 17. Integrate

Function `def poly_integral(poly, C=0):` that calculates the integral of a polynomial. Where `C` is the integration constant.

Example:

f(x) = x³ + 3x + 5 

↓         ↓      ↓

f(x) = 5 + 3x  + x³ -> `poly = [5, 3, 0, 1]`

∫f(x) ---> ∫(5 + 3x + x³)dx -> C + 5x + (3/2)x² + (1/4)x⁴

```python
>>> poly = [5, 3, 0, 1]
>>> print(poly_integral(poly))
[0, 5, 1.5, 0, 0.25]
>>> poly = [5, 3, 0, 1]
>>> print(poly_integral(poly, 1))
[1, 5, 1.5, 0, 0.25]
```
 
