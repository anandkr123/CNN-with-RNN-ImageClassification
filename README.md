# CNN-with-RNN-ImageClassification


1. COIL-100 dataset Challenge is a Kaggle Challenge. Image are rotated from 0-360 degress. 7,200
   images of 100 objects-72 poses per object. Image(128*128) is divided into 128 horizontal pixel rows
   that serves as input at continuous 128 time-stamps to a LSTM network. Further this similar approach
    is performed vertically to make the model see both the structures.

2. Also used a CNN+LSTM model where for every class, where for every one of the 72 poses per
   object, created a representation of that image using a CNN. Each of the 72 representation(in order of
   increasing angle of rotation) is fed sequentially to a LSTM network. This models had a better
   prediction accuracy for different objects compared to a single LSTM based approach.
      
(Used only 15 object with 21 poses per object)

LOSS, ACCURACY, TEST IMAGES PREDICTION 

![Loss_Acc_Pred](https://user-images.githubusercontent.com/23450113/58441401-17b1c000-80e2-11e9-9d00-bdd5dcd91056.png)
