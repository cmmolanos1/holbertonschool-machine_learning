# Probability

### 0. Initialize Poisson

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

### 1. Poisson PMF 

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