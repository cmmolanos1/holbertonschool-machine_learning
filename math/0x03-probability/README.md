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