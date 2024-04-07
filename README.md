# DL Assignment 2
<h3>Training a CNN model to classify the image dataset ( Inaturalist ) </h3>
<p>The task is to classify given images to a certain class and the given image can be of RGB or gray scale </p>
<p>The challenge with such dataset is that the computation is too high as the image dimension increases and the if the network is deep</p>
<p>The CNN mainly comprises of convolution layer with pooling layer which plays the key role in reducing the computation </p>

<h3>Convolutional Neural Network </h3>
<p>Like any other deep neural network, CNN to has layer of neurons associated with an activation function but the key difference is that as the layers go by the dimension of the feature space reduces</p>
<p>Mainly because of the convolution layers and the pooling layers which use kernels ( filters ) to perform convolution on each submatrix to generate certain value to represent the whole submatrix information.</p>
<p>The scale as which the feature space reduces depends on the kernel size used, stride and the padding used for the convolution operation. Its more like a sliding matrix that covers whole matrix to generate a new feature instance from the old feature information.</p>
