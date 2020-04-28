# Probability

Learning:

- Mean.
- Median.
- Mode.
- Probability.
- Variance and Standard Deviation.
- Binomial Distribution.
- Poisson Distibution.
- Exponential Distribution.
- Normal Distribution.

## Tasks

#### 0. Initialize Poisson

File: `poisson.py`

Create a class Poisson that represents a poisson distribution.

* Class constructor `def __init__(self, data=None, lambtha=1.)`

Example:
```python
>>> import numpy as np
>>> from poisson import Poisson
>>> np.random.seed(0)
>>> data = np.random.poisson(5., 100).tolist()
>>> p1 = Poisson(data)
>>> print('Lambtha:', p1.lambtha)
Lambtha: 4.84
>>> p2 = Poisson(lambtha=5)
print('Lambtha:', p2.lambtha)
Lambtha: 5.0
```

#### 1. Poisson PMF 

File: `poisson.py`

Create the instance method to calculate the Probability Mass Function(f(x;λ)) `def pmf(self, k):`, where `k` is the number of “successes”.

```python
>>> import numpy as np
>>> from poisson import Poisson
>>> np.random.seed(0)
>>> data = np.random.poisson(5., 100).tolist()
>>> p1 = Poisson(data)
>>> print('P(9):', p1.pmf(9))
P(9): 0.03175849616802446
>>> p2 = Poisson(lambtha=5)
>>> print('P(9):', p2.pmf(9))
P(9): 0.036265577412911795
```

#### 2. Poisson CDF 

File: `poisson.py`

Create the instance method to calculate the cumulative distribution function `def cdf(self, k):`, where `k` is the number of “successes”.

```python
>>> import numpy as np
>>> from poisson import Poisson
>>> np.random.seed(0)
>>> data = np.random.poisson(5., 100).tolist()
>>> p1 = Poisson(data)
>>> print('F(9):', p1.cdf(9))
F(9): 0.9736102067423525
>>> p2 = Poisson(lambtha=5)
>>> print('F(9):', p2.cdf(9))
F(9): 0.9681719426208609
```

#### 3. Initialize Exponential

File: `exponential.py`

Create a class Exponential that represents a exponential distribution.

* Class constructor `def __init__(self, data=None, lambtha=1.):`

Example:
```python
>>> import numpy as np
>>> from exponential import Exponential
>>> np.random.seed(0)
>>> data = np.random.exponential(0.5, 100).tolist()
>>> e1 = Exponential(data)
>>> print('Lambtha:', e1.lambtha)
Lambtha: 2.1771114730906937
>>> e2 = Exponential(lambtha=2)
>>> print('Lambtha:', e2.lambtha)
Lambtha: 2.0
```

#### 4. Exponential PDF 

File: `exponential.py`

Create the instance method to calculate the Probability Density Function of the distribution.

Example:

```python
>>> import numpy as np
>>> from exponential import Exponential
>>> np.random.seed(0)
>>> data = np.random.exponential(0.5, 100).tolist()
>>> e1 = Exponential(data)
>>> print('f(1):', e1.pdf(1))
f(1): 0.24681591903431568
>>> e2 = Exponential(lambtha=2)
>>> print('f(1):', e2.pdf(1))
f(1): 0.2706705664650693
```

#### 5. Exponential CDF

File: `exponential.py`

Create the instance method to calculate the cumulative distribution function.

Example:

```python
>>> import numpy as np
>>> from exponential import Exponential
>>> np.random.seed(0)
>>> data = np.random.exponential(0.5, 100).tolist()
>>> e1 = Exponential(data)
>>> print('F(1):', e1.cdf(1))
F(1): 0.886631473819791
>>> e2 = Exponential(lambtha=2)
>>> print('F(1):', e2.cdf(1))
F(1): 0.8646647167674654
```

#### 6. Initialize Normal 

File: `normal.py`

Create a class Normal that represents a normal distribution.

* Class constructor `def __init__(self, data=None, mean=0., stddev=1.):`

Example:

```python
>>> import numpy as np
>>> from normal import Normal
>>> np.random.seed(0)
>>> data = np.random.normal(70, 10, 100).tolist()
>>> n1 = Normal(data)
>>> print('Mean:', n1.mean, ', Stddev:', n1.stddev)
Mean: 70.59808015534485 , Stddev: 10.078822447165797
>>> n2 = Normal(mean=70, stddev=10)
>>> print('Mean:', n2.mean, ', Stddev:', n2.stddev)
Mean: 70.0 , Stddev: 10.0
```

#### 7. Normalize Normal 

File: `normal.py`

Create instance methods to calculate z from x, and x from z.

Example:

```python
>>> import numpy as np
>>> from normal import Normal
>>> np.random.seed(0)
>>> data = np.random.normal(70, 10, 100).tolist()
>>> n1 = Normal(data)
>>> print('Z(90):', n1.z_score(90))
Z(90): 1.9250185174272068
>>> print('X(2):', n1.x_value(2))
X(2): 90.75572504967644
>>> n2 = Normal(mean=70, stddev=10)
>>> print('Z(90):', n2.z_score(90))
Z(90): 2.0
>>> print('X(2):', n2.x_value(2))
X(2): 90.0
```

#### 8. Normal PDF

File: `normal.py`

Create the instance method to calculate the Probability Density Function of the distribution.

Example:

```python
>>> import numpy as np
>>> from normal import Normal
>>> np.random.seed(0)
>>> data = np.random.normal(70, 10, 100).tolist()
>>> n1 = Normal(data)
>>> print('PSI(90):', n1.pdf(90))
PSI(90): 0.006206096804434349
>>> n2 = Normal(mean=70, stddev=10)
>>> print('PSI(90):', n2.pdf(90))
PSI(90): 0.005399096651147344
```

#### 9. Normal CDF

File: `normal.py`

Create the instance method to calculate the cumulative distribution function.

Example:

```python
>>> import numpy as np
>>> from normal import Normal
>>> np.random.seed(0)
>>> data = np.random.normal(70, 10, 100).tolist()
>>> n1 = Normal(data)
>>> print('PHI(90):', n1.cdf(90))
PHI(90): 0.982902011086006
>>> n2 = Normal(mean=70, stddev=10)
>>> print('PHI(90):', n2.cdf(90))
PHI(90): 0.9922398930667251
```

#### 10. Initialize Binomial

File: `binomial.py`

Create a class Binomial that represents a binomial distribution.

* Class constructor `def __init__(self, data=None, n=1, p=0.5):`

Example:

```python
>>> import numpy as np
>>> from binomial import Binomial
>>> np.random.seed(0)
>>> data = np.random.binomial(50, 0.6, 100).tolist()
>>> b1 = Binomial(data)
>>> print('n:', b1.n, "p:", b1.p)
n: 50 p: 0.606
>>> b2 = Binomial(n=50, p=0.6)
>>> print('n:', b2.n, "p:", b2.p)
n: 50 p: 0.6
```

#### 11. Binomial PMF

File: `binomial.py`

Create the instance method to calculate the Probability Mass Function.

Example:

```python
>>> import numpy as np
>>> from binomial import Binomial
>>> np.random.seed(0)
>>> data = np.random.binomial(50, 0.6, 100).tolist()
>>> b1 = Binomial(data)
>>> print('P(30):', b1.pmf(30))
P(30): 0.11412829839570347
>>> b2 = Binomial(n=50, p=0.6)
>>> print('P(30):', b2.pmf(30))
P(30): 0.114558552829524
```

#### Binomial CDF 

File: `binomial.py`

Create the instance method to calculate the cumulative distribution function.

Example:

```python
>>> import numpy as np
>>> from binomial import Binomial
>>> np.random.seed(0)
>>> data = np.random.binomial(50, 0.6, 100).tolist()
>>> b1 = Binomial(data)
>>> print('F(30):', b1.cdf(30))
F(30): 0.5189392017296368
>>> b2 = Binomial(n=50, p=0.6)
>>> print('F(30):', b2.cdf(30))
F(30): 0.5535236207894576
```