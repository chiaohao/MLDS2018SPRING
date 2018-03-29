## 1-2.1
train:
```
cd 1/
python3 mnist_dnn_1-2-1.py
```
visualize:
```
python3 mnist_vis_op.py
```
output image: output1-2.1_mnist.jpg

## 1-2.2
train:
```
cd 2/
python3 mnist_dnn_1-2-2.py
python3 simple_dnn_1-2-2.py
```
visualize:
```
python3 visualize.py g_norm_mnist.npy loss_mnist.npy MNIST
python3 visualize.py g_norm_simple.npy loss_simple.npy Simple
```
output images: MNIST_grad_norm.png MNIST_loss.png Simple_grad_norm.png Simple_loss.png

## 1-2.3
train and visualize:
```
cd 3/
python3 simple_dnn_1-2-3.py
```

