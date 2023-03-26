# Classification-Using-CNN
 In this project ,I build and trained a small convolutional neural network for classification task as part of my coursework.
 ## Dataset
 The Dataset is the SVHN which consisited of printed digits 7 and 8,cropped from the house plate pictures . There are a total of 73,257 training samples and 26,032 testing samples. The input image resolution is 32x32 and consists of 3 (RGB) channels.you can find more about the dataset in here.http://ufldl.stanford.edu/housenumbers/

## METHODOLOGY
 ### DATA PRE-PROCESSING
  Initially the data is normalised and as a result the pixel values in all the three
 channels (RGB)are in the range of 0-1.The labels are encoded using one-hot vector
 encoding.
 ### CNN MODEL
  The model consists of 3 convolution layers and each layer has Relu as an activation
function.There are 2 max pooling layers and 3 fully connected layers.The first two
fully connected layers use Relu as the activation function,but the last layers with the
10 nodes use softmax as an activation function
The CNN architecture and parameter settings are as follows: 
![](Screenshot(244).png)
For all the convolutional layers, zero padding is used to get the output of the same
size.SGD (Stochastic Gradient Descent) optimizer is used with a learning rate of
0.01 for 24 epochs .Training set is used for training the model and test set used as
the validation set.The test set is used for finding the classification accuracy of the
model
