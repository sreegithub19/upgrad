# Problem Statement

> Imagine you are working as a data scientist at a home electronics company which manufactures state-of-the-art smart televisions. You want to develop a cool feature in the smart-TV that can recognise five different gestures performed by the user which will help users control the TV without using a remote.
> The gestures are continuously monitored by the webcam mounted on the TV. Each gesture corresponds to a specific command:

- Thumbs up: Increase the volume
- Thumbs down: Decrease the volume
- Left swipe: 'Jump' backwards 10 seconds
- Right swipe: 'Jump' forward 10 seconds
- Stop: Pause the movie
- Each video is a sequence of 30 frames (or images)

# Understanding the Dataset

- The training data consists of a few hundred videos categorised into one of the five classes. Each video (typically 2-3 seconds long) is divided into a sequence of 30 frames(images). These videos have been recorded by various people performing one of the five gestures in front of a webcam similar to what the smart TV will use.

- The data is in a zip file. The zip file contains a 'train' and a 'val' folder with two CSV files for the two folders.

# Objective

- Our task is to train different models on the 'train' folder to predict the action performed in each sequence or video and which performs well on the 'val' folder as well. The final test folder for evaluation is withheld - final model's performance will be tested on the 'test' set.

- Two types of architectures suggested for analyzing videos using deep learning:

## Model Description:

> ### 2. CNN + RNN architecture

- The conv2D network will extract a feature vector for each image, and a sequence of these feature vectors is then fed to an RNN-based network. The output of the RNN is a regular softmax (for a classification problem such as this one).

> ### 1. 3D Convolutional Neural Networks (Conv3D)

- 3D convolutions are a natural extension to the 2D convolutions you are already familiar with.Just like in 2D conv, you move the filter in two directions (x and y), in 3D conv, you move the filter in three directions (x, y and z). In this case, the input to a 3D conv is a video (which is a sequence of 30 RGB images). If we assume that the shape of each image is 100 x 100 x 3, for example, the video becomes a 4D tensor of shape 100 x 100 x 3 x 30 which can be written as (100 x 100 x 30) x 3 where 3 is the number of channels. Hence, deriving the analogy from 2D convolutions where a 2D kernel/filter (a square filter) is represented as (f x f) x c where f is filter size and c is the number of channels, a 3D kernel/filter (a 'cubic' filter) is represented as (f x f x f)
  x c (here c = 3 since the input images have three channels). This cubic filter will now '3Dconvolve' on each of the three channels of the (100 x 100 x 30) tensor.

> Data Generator
> This is one of the most important parts of the code. In the generator, we are going to preprocess the images as we have images of different dimensions (50 x 50, 70 x 70 and 120 x 120) as well as create a batch of video frames. The generator should be able to take a batch of videos as input without any error. Steps like cropping/resizing and normalization should be performed successfully.

> Data Pre-processing

- Resizing: This was mainly done to ensure that the NN only recognizes the gestures effectively. 
- •	Normalization of the images: Normalizing the RGB values of an image can at times be a simple and effective way to get rid of distortions caused by lights and shadows in an image.

> NN Architecture development and training

- - Experimented with different model configurations and hyper-parameters and various iterations and combinations of batch sizes, image dimensions, filter sizes, padding and stride length. We also played around with different learning rates and ReduceLROnPlateau was used to decrease the learning rate if the monitored metrics (val_loss) remains unchanged in between epochs. 

- - We experimented with SGD() and Adam() optimizers but went forward with SGD as it lead to improvement in model’s accuracy by rectifying high variance in the model’s parameters. Played with multiple parameters of the SGD like decay_rate, starting learning rate.

- - We also made use of Batch Normalization, pooling, and dropout layers when our model started to overfit, this could be easily witnessed when our model started giving poor validation accuracy in spite of having good training accuracy.  

- - Early stopping was used to put a halt at the training process when the val_loss would start to saturate / model’s performance would stop improving.

>  Observations

- - It was observed that as the Number of trainable parameters increase, the model takes much more time for training. 
- - Batch size Vs GPU memory: A large batch size can throw GPU Out of memory error (eg: Model-1 has batch size of 64), and thus here we had to play around with the batch size till we were able to arrive at an optimal value of the batch size which our GPU could support (RTX 5000 in Jarvis Labs). 
- - We also found out that the middle frames gives us most of the information and because the train images were chosen so carefully, data augmentation was not required though left-right flipping and zoom, slight rotation could have been done.
- - Increasing the batch size leads to decrease in the training time but this also has a negative impact on the model accuracy. This made us realise that there is always a trade-off here on basis of priority. If we want our model to be ready in a shorter time span, choose larger batch size or for more accuracy we can choose smaller batch size.
- - Conv3D had better performance than CNN2D+LSTM based model with GRU cells. As per our understanding, this is something which depends on the kind of data we used, the architecture we developed and the hyper-parameters we chose. 
- - Transfer learning boosted the overall accuracy of the model. We made use of the MobileNet Architecture due to its light-weight design and high-speed performance coupled with low maintenance as compared to other well-known architectures like VGG16, AlexNet, GoogleNet etc.  

## Conclusion: 
Transfer learning model worked best for us with all the layers trainable, we can see the conv3d played well from the time distributed one which are enough to test on the image set.

- # Model Statistics

- # Conv3D

- Model 1 : No of Epochs = 15 , batch_size = 64 ,shape = (120,120) , no of frames = 10
- - - - Model 1 is giving the out of memory error with batch size 64. We try with less batch size and shapes to further improve the performance and accuracy

- Model 2 : No of Epochs = 20 , batch_size = 20 ,shape = (50,50) , no of frames = 6

- - - - Training Accuracy : 95.74% , Validation Accuracy : 89% ,
- - - - Model Analysis : Training and validation Accuracy are good so that we can conclude that with above set of parameters model is giving good results

- Model 3 : No of Epochs = 20 , batch_size = 30 ,shape = (50,50) , no of frames = 10

- - - - Training Accuracy : 95.29% , Validation Accuracy : 87%
- - - - Model Analysis : Keeping the same shape and increasing the number of frames we have observed that validation accuracy decreased and seems to be overfitting as compared to Model-2

- Model 4 : No of Epochs = 25 , batch_size = 50 ,shape = (100,100) , no of frames = 10

- - - - Training Accuracy : 91.71% , Validation Accuracy : 86%
- - - - Model Analysis : Increasing the image size decreases the accuracy. Also, this model seems to be overfitting.

- Model 5 : No of Epochs = 25 , Batch_size = 50 , shape = (70,70) , no of frames = 18

- - - - Training Accuracy : 95.71% , Validation Accuracy : 87%
- - - - Model Analysis : This model is clearly an overfit model can see that increasing in number of frames and epochs causing the noise to be learned also from all the frames

- # CNN + RNN : CNN2D LSTM Model - TimeDistributed

- Model 6 : No of Epochs = 25 , Batch_size = 50 , shape = (70,70) , no of frames = 18

- - - - Training Accuracy : 81.79% , Validation Accuracy : 60%
- - - - Model Analysis : This model is clearly Overfitting

- Model 7 : No of epochs = 20 , batch_size = 20 , shape (50,50) , no of frames = 10

- - - - Training Accuracy : 84.71% , Validation Accuracy : 67%
- - - - Model Analysis : This model is clearly overfitting

- # CONV2D + GRU

- Model 8 : No of epochs = 20 , batch_size = 20 , shape (50,50) , no of frames = 18

- - - - Training Accuracy : 94.26%, Validation Accuracy : 72%
- - - - Model Analysis : This model is overfitting

- # Transfer Learning Using MobileNet

- Model 9 : No of epochs = 15 , batch_size = 5 , shape (120,120) , no of frames = 18

- - - - Training Accuracy : 99.55% , Validation Accuracy : 95%
- - - - Model Analysis : This is so far the best model that we got with better accuracy
