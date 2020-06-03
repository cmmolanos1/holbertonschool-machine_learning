# Convolution

Coding from scratch the convolution and pooling methods.

### Tasks

0\. Valid Convolution mandatory

Write a function `def convolve_grayscale_valid(images, kernel):` that performs a valid convolution on grayscale images:

    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 0-main.py 
    #!/usr/bin/env python3
    
    import matplotlib.pyplot as plt
    import numpy as np
    convolve_grayscale_valid = __import__('0-convolve_grayscale_valid').convolve_grayscale_valid
    
    
    if __name__ == '__main__':
    
        dataset = np.load('../../supervised_learning/data/MNIST.npz')
        images = dataset['X_train']
        print(images.shape)
        kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
        images_conv = convolve_grayscale_valid(images, kernel)
        print(images_conv.shape)
    
        plt.imshow(images[0], cmap='gray')
        plt.show()
        plt.imshow(images_conv[0], cmap='gray')
        plt.show()
    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./0-main.py 
    (50000, 28, 28)
    (50000, 26, 26)
    
![]()

#### 1\. Same Convolution mandatory

Write a function `def convolve_grayscale_same(images, kernel):` that performs a same convolution on grayscale images:

    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 1-main.py 
    #!/usr/bin/env python3
    
    import matplotlib.pyplot as plt
    import numpy as np
    convolve_grayscale_same = __import__('1-convolve_grayscale_same').convolve_grayscale_same
    
    
    if __name__ == '__main__':
    
        dataset = np.load('../../supervised_learning/data/MNIST.npz')
        images = dataset['X_train']
        print(images.shape)
        kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
        images_conv = convolve_grayscale_same(images, kernel)
        print(images_conv.shape)
    
        plt.imshow(images[0], cmap='gray')
        plt.show()
        plt.imshow(images_conv[0], cmap='gray')
        plt.show()
    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./1-main.py 
    (50000, 28, 28)
    (50000, 28, 28)
    


#### 2\. Convolution with Padding mandatory

Write a function `def convolve_grayscale_padding(images, kernel, padding):` that performs a convolution on grayscale images with custom padding:

    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 2-main.py 
    #!/usr/bin/env python3
    
    import matplotlib.pyplot as plt
    import numpy as np
    convolve_grayscale_padding = __import__('2-convolve_grayscale_padding').convolve_grayscale_padding
    
    
    if __name__ == '__main__':
    
        dataset = np.load('../../supervised_learning/data/MNIST.npz')
        images = dataset['X_train']
        print(images.shape)
        kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
        images_conv = convolve_grayscale_padding(images, kernel, (2, 4))
        print(images_conv.shape)
    
        plt.imshow(images[0], cmap='gray')
        plt.show()
        plt.imshow(images_conv[0], cmap='gray')
        plt.show()
    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./2-main.py 
    (50000, 28, 28)
    (50000, 30, 34)
    


#### 3\. Strided Convolution mandatory

Write a function `def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):` that performs a convolution on grayscale images:

    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 3-main.py 
    #!/usr/bin/env python3
    
    import matplotlib.pyplot as plt
    import numpy as np
    convolve_grayscale = __import__('3-convolve_grayscale').convolve_grayscale
    
    
    if __name__ == '__main__':
    
        dataset = np.load('../../supervised_learning/data/MNIST.npz')
        images = dataset['X_train']
        print(images.shape)
        kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
        images_conv = convolve_grayscale(images, kernel, padding='valid', stride=(2, 2))
        print(images_conv.shape)
    
        plt.imshow(images[0], cmap='gray')
        plt.show()
        plt.imshow(images_conv[0], cmap='gray')
        plt.show()
    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./3-main.py 
    (50000, 28, 28)
    (50000, 13, 13)
    


#### 4\. Convolution with Channels mandatory

Write a function `def convolve_channels(images, kernel, padding='same', stride=(1, 1)):` that performs a convolution on images with channels:

    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 4-main.py 
    #!/usr/bin/env python3
    
    import matplotlib.pyplot as plt
    import numpy as np
    convolve_channels = __import__('4-convolve_channels').convolve_channels
    
    
    if __name__ == '__main__':
    
        dataset = np.load('../../supervised_learning/data/animals_1.npz')
        images = dataset['data']
        print(images.shape)
        kernel = np.array([[[0, 0, 0], [-1, -1, -1], [0, 0, 0]], [[-1, -1, -1], [5, 5, 5], [-1, -1, -1]], [[0, 0, 0], [-1, -1, -1], [0, 0, 0]]])
        images_conv = convolve_channels(images, kernel, padding='valid')
        print(images_conv.shape)
    
        plt.imshow(images[0])
        plt.show()
        plt.imshow(images_conv[0])
        plt.show()
    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./4-main.py 
    (10000, 32, 32, 3)
    (10000, 30, 30)
    


#### 5\. Multiple Kernels mandatory

Write a function `def convolve(images, kernels, padding='same', stride=(1, 1)):` that performs a convolution on images using multiple kernels:

    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 5-main.py 
    #!/usr/bin/env python3
    
    import matplotlib.pyplot as plt
    import numpy as np
    convolve = __import__('5-convolve').convolve
    
    
    if __name__ == '__main__':
    
        dataset = np.load('../../supervised_learning/data/animals_1.npz')
        images = dataset['data']
        print(images.shape)
        kernels = np.array([[[[0, 1, 1], [0, 1, 1], [0, 1, 1]], [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], [[0, -1, 1], [0, -1, 1], [0, -1, 1]]],
                           [[[-1, 1, 0], [-1, 1, 0], [-1, 1, 0]], [[5, 0, 0], [5, 0, 0], [5, 0, 0]], [[-1, -1, 0], [-1, -1, 0], [-1, -1, 0]]],
                           [[[0, 1, -1], [0, 1, -1], [0, 1, -1]], [[-1, 0, -1], [-1, 0, -1], [-1, 0, -1]], [[0, -1, -1], [0, -1, -1], [0, -1, -1]]]])
    
        images_conv = convolve(images, kernels, padding='valid')
        print(images_conv.shape)
    
        plt.imshow(images[0])
        plt.show()
        plt.imshow(images_conv[0, :, :, 0])
        plt.show()
        plt.imshow(images_conv[0, :, :, 1])
        plt.show()
        plt.imshow(images_conv[0, :, :, 2])
        plt.show()
    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./5-main.py 
    (10000, 32, 32, 3)
    (10000, 30, 30, 3)
    


#### 6\. Pooling mandatory

Write a function `def pool(images, kernel_shape, stride, mode='max'):` that performs pooling on images:

    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 6-main.py 
    #!/usr/bin/env python3
    
    import matplotlib.pyplot as plt
    import numpy as np
    pool = __import__('6-pool').pool
    
    
    if __name__ == '__main__':
    
        dataset = np.load('../../supervised_learning/data/animals_1.npz')
        images = dataset['data']
        print(images.shape)
        images_pool = pool(images, (2, 2), (2, 2), mode='avg')
        print(images_pool.shape)
    
        plt.imshow(images[0])
        plt.show()
        plt.imshow(images_pool[0] / 255)
        plt.show()
    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./6-main.py 
    (10000, 32, 32, 3)
    (10000, 16, 16, 3)
    

