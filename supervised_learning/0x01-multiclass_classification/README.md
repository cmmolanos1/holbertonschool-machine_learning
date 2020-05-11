# Multiclass Classification

## 0. One-Hot Encode

Write a function def one_hot_encode(Y, classes): that converts a numeric label vector into a one-hot matrix.

### Example:

```
user@ubuntu-xenial:0x01-multiclass_classification$ cat 0-main.py
#!/usr/bin/env python3

import numpy as np

oh_encode = __import__('0-one_hot_encode').one_hot_encode

lib = np.load('../data/MNIST.npz')
Y = lib['Y_train'][:10]

print(Y)
Y_one_hot = oh_encode(Y, 10)
print(Y_one_hot)
user@ubuntu-xenial:0x01-multiclass_classification$ ./0-main.py
[5 0 4 1 9 2 1 3 1 4]
[[0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 1. 0. 1. 0.]
 [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]]
user@ubuntu-xenial:0x01-multiclass_classification$
```

## 1. One-Hot Decode

Write a function def one_hot_decode(one_hot): that converts a one-hot matrix into a vector of labels.

### Example:

```
user@ubuntu-xenial:0x01-multiclass_classification$ cat 1-main.py
#!/usr/bin/env python3

import numpy as np

oh_encode = __import__('0-one_hot_encode').one_hot_encode
oh_decode = __import__('1-one_hot_decode').one_hot_decode

lib = np.load('../data/MNIST.npz')
Y = lib['Y_train'][:10]

print(Y)
Y_one_hot = oh_encode(Y, 10)
Y_decoded = oh_decode(Y_one_hot)
print(Y_decoded)
user@ubuntu-xenial:0x01-multiclass_classification$ ./1-main.py
[5 0 4 1 9 2 1 3 1 4]
[5 0 4 1 9 2 1 3 1 4]
user@ubuntu-xenial:0x01-multiclass_classification$
```

## 2. Persistence is Key

Create methods save and load to create and read pickle files.

## 3. Update DeepNeuralNetwork

Modifies the DNN. The last layer is softmax, evaluate and cost are modified.

## 4. All the Activations

Modifies the activation of hidden layers between sigmoid or tanh.
