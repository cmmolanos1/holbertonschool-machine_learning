# Tensorflow

First steps coding this framework.

## 0. Placeholders

Write the function `def create_placeholders(nx, classes):` that returns two placeholders, x and y, for the neural network.

```
ubuntu@alexa-ml:~/0x02-tensorflow$ cat 0-main.py 
#!/usr/bin/env python3

import tensorflow as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders

x, y = create_placeholders(784, 10)
print(x)
print(y)
ubuntu@alexa-ml:~/0x02-tensorflow$ ./0-main.py 
Tensor("x:0", shape=(?, 784), dtype=float32)
Tensor("y:0", shape=(?, 10), dtype=float32)
ubuntu@alexa-ml:~/0x02-tensorflow$ 
```
## 1. Layers

Write the function `def create_layer(prev, n, activation):`

```
ubuntu@alexa-ml:~/0x02-tensorflow$ cat 1-main.py 
#!/usr/bin/env python3

import tensorflow as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders
create_layer = __import__('1-create_layer').create_layer

x, y = create_placeholders(784, 10)
l = create_layer(x, 256, tf.nn.tanh)
print(l)
ubuntu@alexa-ml:~/0x02-tensorflow$ ./1-main.py 
Tensor("layer/Tanh:0", shape=(?, 256), dtype=float32)
ubuntu@alexa-ml:~/0x02-tensorflow$ 
```

## 2. Forward Propagation

Write the function `def forward_prop(x, layer_sizes=[], activations=[]):` that creates the forward propagation graph for the neural network:

```
ubuntu@alexa-ml:~/0x02-tensorflow$ cat 2-main.py 
#!/usr/bin/env python3

import tensorflow as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop

x, y = create_placeholders(784, 10)
y_pred = forward_prop(x, [256, 256, 10], [tf.nn.tanh, tf.nn.tanh, None])
print(y_pred)
ubuntu@alexa-ml:~/0x02-tensorflow$ ./2-main.py 
Tensor("layer_2/BiasAdd:0", shape=(?, 10), dtype=float32)
ubuntu@alexa-ml:~/0x02-tensorflow$ 
```

## 3. Accuracy 

Write the function `def calculate_accuracy(y, y_pred):` that calculates the accuracy of a prediction:

```
ubuntu@alexa-ml:~/0x02-tensorflow$ cat 3-main.py 
#!/usr/bin/env python3

import tensorflow as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy

x, y = create_placeholders(784, 10)
y_pred = forward_prop(x, [256, 256, 10], [tf.nn.tanh, tf.nn.tanh, None])
accuracy = calculate_accuracy(y, y_pred)
print(accuracy)
ubuntu@alexa-ml:~/0x02-tensorflow$ ./3-main.py 
Tensor("Mean:0", shape=(), dtype=float32)
ubuntu@alexa-ml:~/0x02-tensorflow$
```

## Loss

Write the function `def calculate_loss(y, y_pred):` that calculates the softmax cross-entropy loss of a prediction:

```
ubuntu@alexa-ml:~/0x02-tensorflow$ cat 4-main.py 
#!/usr/bin/env python3

import tensorflow as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_loss = __import__('4-calculate_loss').calculate_loss

x, y = create_placeholders(784, 10)
y_pred = forward_prop(x, [256, 256, 10], [tf.nn.tanh, tf.nn.tanh, None])
loss = calculate_loss(y, y_pred)
print(loss)
ubuntu@alexa-ml:~/0x02-tensorflow$ ./4-main.py 
Tensor("softmax_cross_entropy_loss/value:0", shape=(), dtype=float32)
ubuntu@alexa-ml:~/0x02-tensorflow$ 
```

## 5. Train_Op

Write the function def create_train_op(loss, alpha): that creates the training operation for the network:

```
ubuntu@alexa-ml:~/0x02-tensorflow$ cat 5-main.py 
#!/usr/bin/env python3

import tensorflow as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


x, y = create_placeholders(784, 10)
y_pred = forward_prop(x, [256, 256, 10], [tf.nn.tanh, tf.nn.tanh, None])
loss = calculate_loss(y, y_pred)
train_op = create_train_op(loss, 0.01)
print(train_op)
ubuntu@alexa-ml:~/0x02-tensorflow$ ./5-main.py 
name: "GradientDescent"
op: "NoOp"
input: "^GradientDescent/update_layer/kernel/ApplyGradientDescent"
input: "^GradientDescent/update_layer/bias/ApplyGradientDescent"
input: "^GradientDescent/update_layer_1/kernel/ApplyGradientDescent"
input: "^GradientDescent/update_layer_1/bias/ApplyGradientDescent"
input: "^GradientDescent/update_layer_2/kernel/ApplyGradientDescent"
input: "^GradientDescent/update_layer_2/bias/ApplyGradientDescent"

ubuntu@alexa-ml:~/0x02-tensorflow$ 
```

## 6. Train 

Write the function `def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):` that builds, trains, and saves a neural network classifier:

```
ubuntu@alexa-ml:~/0x02-tensorflow$ cat 6-main.py 
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
train = __import__('6-train').train

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_train_3D = lib['X_train']
    Y_train = lib['Y_train']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1))
    Y_train_oh = one_hot(Y_train, 10)
    X_valid_3D = lib['X_valid']
    Y_valid = lib['Y_valid']
    X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1))
    Y_valid_oh = one_hot(Y_valid, 10)

    layer_sizes = [256, 256, 10]
    activations = [tf.nn.tanh, tf.nn.tanh, None]
    alpha = 0.01
    iterations = 1000

    tf.set_random_seed(0)
    save_path = train(X_train, Y_train_oh, X_valid, Y_valid_oh, layer_sizes,
                      activations, alpha, iterations, save_path="./model.ckpt")
    print("Model saved in path: {}".format(save_path))
ubuntu@alexa-ml:~/0x02-tensorflow$ ./6-main.py 
2018-11-03 01:04:55.281078: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
After 0 iterations:
    Training Cost: 2.8232274055480957
    Training Accuracy: 0.08726000040769577
    Validation Cost: 2.810533285140991
    Validation Accuracy: 0.08640000224113464
After 100 iterations:
    Training Cost: 0.8393384218215942
    Training Accuracy: 0.7824000120162964
    Validation Cost: 0.7826032042503357
    Validation Accuracy: 0.8061000108718872
After 200 iterations:
    Training Cost: 0.6094841361045837
    Training Accuracy: 0.8396000266075134
    Validation Cost: 0.5562412142753601
    Validation Accuracy: 0.8597999811172485

...

After 1000 iterations:
    Training Cost: 0.352960467338562
    Training Accuracy: 0.9004999995231628
    Validation Cost: 0.32148978114128113
    Validation Accuracy: 0.909600019454956
Model saved in path: ./model.ckpt
ubuntu@alexa-ml:~/0x02-tensorflow$ ls model.ckpt*
model.ckpt.data-00000-of-00001  model.ckpt.index  model.ckpt.meta
ubuntu@alexa-ml:~/0x02-tensorflow$

```

## 7. Evaluate

Write the function `def evaluate(X, Y, save_path):` that evaluates the output of a neural network:

```
ubuntu@alexa-ml:~/0x02-tensorflow$ cat 7-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
evaluate = __import__('7-evaluate').evaluate

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_test_3D = lib['X_test']
    Y_test = lib['Y_test']
    X_test = X_test_3D.reshape((X_test_3D.shape[0], -1))
    Y_test_oh = one_hot(Y_test, 10)

    Y_pred_oh, accuracy, cost = evaluate(X_test, Y_test_oh, './model.ckpt')
    print("Test Accuracy:", accuracy)
    print("Test Cost:", cost)

    Y_pred = np.argmax(Y_pred_oh, axis=1)

    fig = plt.figure(figsize=(10, 10))
    for i in range(100):
        fig.add_subplot(10, 10, i + 1)
        plt.imshow(X_test_3D[i])
        plt.title(str(Y_test[i]) + ' : ' + str(Y_pred[i]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
ubuntu@alexa-ml:~/0x02-tensorflow$ ./7-main.py
2018-11-03 02:08:30.767168: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
Test Accuracy: 0.9391
Test Cost: 0.21756475
```