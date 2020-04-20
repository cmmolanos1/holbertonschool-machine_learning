#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))
index = ('Farrah', 'Fred', 'Felicia')
fruits = ('apples', 'bananas', 'oranges', 'peaches')
colors = ('red', 'yellow', '#ff8000', '#ffe5b4')
width = 0.5

for i in range(fruit.shape[0]):
    plt.bar(index, fruit[i], width,
            bottom=np.sum(fruit[:i], axis=0),
            color=colors[i],
            label=fruits[i])

plt.ylabel('Quantity of Fruit')
plt.yticks(np.arange(0, 90, step=10))
plt.title("Number of Fruit per Person")
plt.legend()
plt.show()
