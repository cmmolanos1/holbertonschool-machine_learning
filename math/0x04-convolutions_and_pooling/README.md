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
    

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/17e3fb852b947ff6d845.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T142109Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=2de6e4e4f1c66b4feeb84eb95be4c68f03b77e41516d2d9f80b5e6abe8b822ae)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/6e1b02cc87497f12f17e.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T142109Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=dc68e091bda88261ad9355a91923279ab616bafa9552c94f663b90d7959c2f58)

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
    

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/17e3fb852b947ff6d845.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T142109Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=2de6e4e4f1c66b4feeb84eb95be4c68f03b77e41516d2d9f80b5e6abe8b822ae)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/b32bba8fea86011c3372.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T142109Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=339a62c40048b8229102de140eacc89bbe6866df83932aa56aa60dc0c938046b)

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
    

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/17e3fb852b947ff6d845.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T142109Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=2de6e4e4f1c66b4feeb84eb95be4c68f03b77e41516d2d9f80b5e6abe8b822ae)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/3f178b675c1e2fdc86bd.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T142109Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=43dac69f4d416bec6c7d0b70b30a85007a1b82f3e19a43fe87f14f26c9584fc5)

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
    

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/17e3fb852b947ff6d845.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T142109Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=2de6e4e4f1c66b4feeb84eb95be4c68f03b77e41516d2d9f80b5e6abe8b822ae)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/036ccba7dccf211dab76.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T142109Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=01fa38c97e025020cb9e19d9f8a4598481cf361daa1fd96cd0968c01454e0ff4)

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
    

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/6add724c812e8dcddb21.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T142109Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=f666580b9255fd2bb30cfb13a5ca0d3ea3c54da7a3ca15acbc9c059376c255f7)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/8bc039fb38d60601b01a.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T142109Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=80fac52cd13f33a7a92ff72914825e9452439c6f3c8e1efbb8614a6003b6b64b)

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
    

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/6add724c812e8dcddb21.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T142109Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=f666580b9255fd2bb30cfb13a5ca0d3ea3c54da7a3ca15acbc9c059376c255f7)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/6d6319bb470e3566e885.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T142109Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=4dd87833ded591208e84610f2fe1ae52acd783bb352bd22599ea6b00cfd3a03e)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/1370dd6200e942eee8f9.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T142109Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=2f434bfd65b03b130c86418d40abd320537fdc49f6eedf3545bec4bb03d027cc)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/a24b7d741b3c378f9f89.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T142109Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=e70e367a4af8be13a25ac0ccabbd5241a738be7209243d559a37dae7ab78862e)

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
    

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/6add724c812e8dcddb21.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T142109Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=f666580b9255fd2bb30cfb13a5ca0d3ea3c54da7a3ca15acbc9c059376c255f7)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/ab4705f939c3a8e487bb.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T142109Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=e4aeedb02abc7e1b6dfdd1c6613069fb87e87bba2f7c3987e2e1e1e09f37b74d)

