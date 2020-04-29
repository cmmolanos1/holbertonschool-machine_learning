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
```
user@ubuntu-xenial:0x03-probability$ cat 0-main.py 
#!/usr/bin/env python3

import numpy as np
Poisson = __import__('poisson').Poisson

np.random.seed(0)
data = np.random.poisson(5., 100).tolist()
p1 = Poisson(data)
print('Lambtha:', p1.lambtha)

p2 = Poisson(lambtha=5)
print('Lambtha:', p2.lambtha)
user@ubuntu-xenial:0x03-probability$ ./0-main.py 
Lambtha: 4.84
Lambtha: 5.0
user@ubuntu-xenial:0x03-probability$
```

#### 1. Poisson PMF 

File: `poisson.py`

Create the instance method to calculate the Probability Mass Function(f(x;λ)) `def pmf(self, k):`, where `k` is the number of “successes”.

```
user@ubuntu-xenial:0x03-probability$ cat 1-main.py 
#!/usr/bin/env python3

import numpy as np
Poisson = __import__('poisson').Poisson

np.random.seed(0)
data = np.random.poisson(5., 100).tolist()
p1 = Poisson(data)
print('P(9):', p1.pmf(9))

p2 = Poisson(lambtha=5)
print('P(9):', p2.pmf(9))
user@ubuntu-xenial:0x03-probability$ ./1-main.py 
P(9): 0.03175849616802446
P(9): 0.036265577412911795
user@ubuntu-xenial:0x03-probability$
```

#### 2. Poisson CDF 

File: `poisson.py`

Create the instance method to calculate the cumulative distribution function `def cdf(self, k):`, where `k` is the number of “successes”.

```
user@ubuntu-xenial:0x03-probability$ cat 2-main.py 
#!/usr/bin/env python3

import numpy as np
Poisson = __import__('poisson').Poisson

np.random.seed(0)
data = np.random.poisson(5., 100).tolist()
p1 = Poisson(data)
print('F(9):', p1.cdf(9))

p2 = Poisson(lambtha=5)
print('F(9):', p2.cdf(9))
user@ubuntu-xenial:0x03-probability$ ./2-main.py 
F(9): 0.9736102067423525
F(9): 0.9681719426208609
user@ubuntu-xenial:0x03-probability$ 
```

#### 3. Initialize Exponential

File: `exponential.py`

Create a class Exponential that represents a exponential distribution.

* Class constructor `def __init__(self, data=None, lambtha=1.):`

Example:
```
user@ubuntu-xenial:0x03-probability$ cat 3-main.py 
#!/usr/bin/env python3

import numpy as np
Exponential = __import__('exponential').Exponential

np.random.seed(0)
data = np.random.exponential(0.5, 100).tolist()
e1 = Exponential(data)
print('Lambtha:', e1.lambtha)

e2 = Exponential(lambtha=2)
print('Lambtha:', e2.lambtha)
user@ubuntu-xenial:0x03-probability$ ./3-main.py 
Lambtha: 2.1771114730906937
Lambtha: 2.0
user@ubuntu-xenial:0x03-probability$
```

#### 4. Exponential PDF 

File: `exponential.py`

Create the instance method to calculate the Probability Density Function of the distribution.

Example:

```
user@ubuntu-xenial:0x03-probability$ cat 4-main.py 
#!/usr/bin/env python3

import numpy as np
Exponential = __import__('exponential').Exponential

np.random.seed(0)
data = np.random.exponential(0.5, 100).tolist()
e1 = Exponential(data)
print('f(1):', e1.pdf(1))

e2 = Exponential(lambtha=2)
print('f(1):', e2.pdf(1))
user@ubuntu-xenial:0x03-probability$ ./4-main.py 
f(1): 0.24681591903431568
f(1): 0.2706705664650693
user@ubuntu-xenial:0x03-probability$
```

#### 5. Exponential CDF

File: `exponential.py`

Create the instance method to calculate the cumulative distribution function.

Example:

```
user@ubuntu-xenial:0x03-probability$ cat 5-main.py 
#!/usr/bin/env python3

import numpy as np
Exponential = __import__('exponential').Exponential

np.random.seed(0)
data = np.random.exponential(0.5, 100).tolist()
e1 = Exponential(data)
print('F(1):', e1.cdf(1))

e2 = Exponential(lambtha=2)
print('F(1):', e2.cdf(1))
user@ubuntu-xenial:0x03-probability$ ./5-main.py 
F(1): 0.886631473819791
F(1): 0.8646647167674654
user@ubuntu-xenial:0x03-probability$
```

#### 6. Initialize Normal 

File: `normal.py`

Create a class Normal that represents a normal distribution.

* Class constructor `def __init__(self, data=None, mean=0., stddev=1.):`

Example:

```
user@ubuntu-xenial:0x03-probability$ cat 6-main.py 
#!/usr/bin/env python3

import numpy as np
Normal = __import__('normal').Normal

np.random.seed(0)
data = np.random.normal(70, 10, 100).tolist()
n1 = Normal(data)
print('Mean:', n1.mean, ', Stddev:', n1.stddev)

n2 = Normal(mean=70, stddev=10)
print('Mean:', n2.mean, ', Stddev:', n2.stddev)
user@ubuntu-xenial:0x03-probability$ ./6-main.py 
Mean: 70.59808015534485 , Stddev: 10.078822447165797
Mean: 70.0 , Stddev: 10.0
user@ubuntu-xenial:0x03-probability$
```

#### 7. Normalize Normal 

File: `normal.py`

Create instance methods to calculate z from x, and x from z.

Example:

```
user@ubuntu-xenial:0x03-probability$ cat 7-main.py 
#!/usr/bin/env python3

import numpy as np
Normal = __import__('normal').Normal

np.random.seed(0)
data = np.random.normal(70, 10, 100).tolist()
n1 = Normal(data)
print('Z(90):', n1.z_score(90))
print('X(2):', n1.x_value(2))

n2 = Normal(mean=70, stddev=10)
print()
print('Z(90):', n2.z_score(90))
print('X(2):', n2.x_value(2))
user@ubuntu-xenial:0x03-probability$ ./7-main.py 
Z(90): 1.9250185174272068
X(2): 90.75572504967644

Z(90): 2.0
X(2): 90.0
user@ubuntu-xenial:0x03-probability$
```

#### 8. Normal PDF

File: `normal.py`

Create the instance method to calculate the Probability Density Function of the distribution.

Example:

```
user@ubuntu-xenial:0x03-probability$ cat 8-main.py 
#!/usr/bin/env python3

import numpy as np
Normal = __import__('normal').Normal

np.random.seed(0)
data = np.random.normal(70, 10, 100).tolist()
n1 = Normal(data)
print('PSI(90):', n1.pdf(90))

n2 = Normal(mean=70, stddev=10)
print('PSI(90):', n2.pdf(90))
user@ubuntu-xenial:0x03-probability$ ./8-main.py 
PSI(90): 0.006206096804434349
PSI(90): 0.005399096651147344
user@ubuntu-xenial:0x03-probability$

```

#### 9. Normal CDF

File: `normal.py`

Create the instance method to calculate the cumulative distribution function.

Example:

```
user@ubuntu-xenial:0x03-probability$ cat 9-main.py 
#!/usr/bin/env python3

import numpy as np
Normal = __import__('normal').Normal

np.random.seed(0)
data = np.random.normal(70, 10, 100).tolist()
n1 = Normal(data)
print('PHI(90):', n1.cdf(90))

n2 = Normal(mean=70, stddev=10)
print('PHI(90):', n2.cdf(90))
user@ubuntu-xenial:0x03-probability$ ./9-main.py 
PHI(90): 0.982902011086006
PHI(90): 0.9922398930667251
user@ubuntu-xenial:0x03-probability$
```

#### 10. Initialize Binomial

File: `binomial.py`

Create a class Binomial that represents a binomial distribution.

* Class constructor `def __init__(self, data=None, n=1, p=0.5):`

Example:

```
user@ubuntu-xenial:0x03-probability$ cat 10-main.py 
#!/usr/bin/env python3

import numpy as np
Binomial = __import__('binomial').Binomial

np.random.seed(0)
data = np.random.binomial(50, 0.6, 100).tolist()
b1 = Binomial(data)
print('n:', b1.n, "p:", b1.p)

b2 = Binomial(n=50, p=0.6)
print('n:', b2.n, "p:", b2.p)
user@ubuntu-xenial:0x03-probability$ ./10-main.py 
n: 50 p: 0.606
n: 50 p: 0.6
user@ubuntu-xenial:0x03-probability$ 
```

#### 11. Binomial PMF

File: `binomial.py`

Create the instance method to calculate the Probability Mass Function.

Example:

```
user@ubuntu-xenial:0x03-probability$ cat 11-main.py 
#!/usr/bin/env python3

import numpy as np
Binomial = __import__('binomial').Binomial

np.random.seed(0)
data = np.random.binomial(50, 0.6, 100).tolist()
b1 = Binomial(data)
print('P(30):', b1.pmf(30))

b2 = Binomial(n=50, p=0.6)
print('P(30):', b2.pmf(30))
user@ubuntu-xenial:0x03-probability$ ./11-main.py 
P(30): 0.11412829839570347
P(30): 0.114558552829524
user@ubuntu-xenial:0x03-probability$
```

#### 12. Binomial CDF 

File: `binomial.py`

Create the instance method to calculate the cumulative distribution function.

Example:

```
user@ubuntu-xenial:0x03-probability$ cat 12-main.py 
#!/usr/bin/env python3

import numpy as np
Binomial = __import__('binomial').Binomial

np.random.seed(0)
data = np.random.binomial(50, 0.6, 100).tolist()
b1 = Binomial(data)
print('F(30):', b1.cdf(30))

b2 = Binomial(n=50, p=0.6)
print('F(30):', b2.cdf(30))
user@ubuntu-xenial:0x03-probability$ ./12-main.py 
F(30): 0.5189392017296368
F(30): 0.5535236207894576
user@ubuntu-xenial:0x03-probability$
```