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
![Screenshot (244)](https://user-images.githubusercontent.com/41173314/227750908-3fb87f53-f397-424c-934b-ac2c7ae858f4.png)
For all the convolutional layers, zero padding is used to get the output of the same
size.SGD (Stochastic Gradient Descent) optimizer is used with a learning rate of
0.01 for 24 epochs .Training set is used for training the model and test set used as
the validation set.The test set is used for finding the classification accuracy of the
model

## Results
Prediction accuracy and loss of the test set is computed.Accuracy is 86% and loss
is 0.526 and also two graphs are plotted for the tensorflow framework model 
The first one is accuracy of the train and test set as function of epochs and is given
below![Screenshot (245)](https://user-images.githubusercontent.com/41173314/227751135-c72aaf79-2603-4b18-98a7-3d5adaa4deef.png)

The second one is loss of the model of both the sets as function of epochs and is
given below ![Screenshot (246)](https://user-images.githubusercontent.com/41173314/227751392-6e082f48-26ba-4154-b121-7ca5ebdfde71.png)

As you can see the accuracy increases and the loss decreases when we increase
the number of epochs for training the model
