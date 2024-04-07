# DL Assignment 2
<h3>Training a CNN model to classify the image dataset ( Inaturalist ) </h3>
<p>The task is to classify given images to a certain class and the given image can be of RGB or gray scale </p>
<p>The challenge with such dataset is that the computation is too high as the image dimension increases and the if the network is deep</p>
<p>The CNN mainly comprises of convolution layer with pooling layer which plays the key role in reducing the computation </p>

<h3>Convolutional Neural Network </h3>
<p>Like any other deep neural network, CNN to has layer of neurons associated with an activation function but the key difference is that as the layers go by the dimension of the feature space reduces</p>
<p>Mainly because of the convolution layers and the pooling layers which use kernels ( filters ) to perform convolution on each submatrix to generate certain value to represent the whole submatrix information.</p>
<p>The scale as which the feature space reduces depends on the kernel size used, stride and the padding used for the convolution operation. Its more like a sliding matrix that covers whole matrix to generate a new feature instance from the old feature information.</p>

<br>
<p>The current CNN model is with 5 convolutional blocks each with a convolution layer, Dropout layer, Maxpool layer</p>
<p>The parameters can be changed for each block using the arguments and the option to enable dropout or disable it and to use batch normalization or not is also incorporated into the model.</p>
<p>After followed by a dense layer which is precedded by flattening of the feature space so that the feedforward layer can be accommodated afterwards.</p>
<p>And an output layer with output neurons to be no of class available (10 in current dataset)</p>

<p>Metrics: Model validation accuracy ~ 35%</p>
<p>Metrics: Model Testing accuracy ~ 40.6%</p>

<h4>Best Model Parameters: </h4>
Activation Function: ELU<br>
Batch Size: 64<br>
Dropout : 0.2 <br>
Epochs : 10<br>
Filter Customization : [32,64,128,256,512]<br>
Kernel Dimensions: [3,3,3,3,3]<br>
Learning Rate : 1e-4<br>
Optimizer : Adam<br>
Weight Decay : 0.0005<br>
Batch Normalization : True<br>
Data Augmentation : True<br>


<h3>Finetuning a pretrained model</h3>
<p>Pretrained models as once that are already rigroously finetuned to certain parameters and are trained on extensive dataset to handle corresponding to certain problem </p>
<p>Consider ResNet or VGGNet models which are initially trained to classify an image into 1000 classes as the imagenet has 1000 classes,but considering the current objective of 10 class problem the model architecture must be altered.</p>
<p>So need to first resize the image to 224 x 224 as the pretrained model was also trained upon the image of dimension 224 x 224 incase of ResNet</p>
<p>The last output layer (Fully connected layer with 1000 output channel and be changed to 10 to accommodate the challenge to classify the images to required classes and also need to train this layer weights by keeping the rest intact during the training process.</p>
<p>Another option would be to freeze the top k layers and train the rest (n-k) layer on the new dataset that is the problem statement for 10 class classification.</p>
<p> Doing so resulted in an accuracy of 79.2% over 10 Epochs</p>
<h4>Configuration: </h4>
Batchsize : 128 <br>
Epochs : 10<br>
Learning rate : 1e-3 <br>
Input Image : 224 x 224 x 3<br>
Dense Layer neurons : 512<br>


