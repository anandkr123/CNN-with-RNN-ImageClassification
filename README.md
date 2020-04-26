# CNN-with-RNN-ImageClassification


1. COIL-100 dataset Challenge is a Kaggle Challenge. Image are rotated from 0-360 degress. 7,200
   images of 100 objects-72 poses per object. Image(128*128) is divided into 128 horizontal pixel rows
   that serves as input at continuous 128 time-stamps to a RNN network.

LOSS, ACCURACY on training and validation set and predictions on test set 

ACCURACY ON TRAINING SET is AROUND 50% while ACCURACY ON VALIDATION SET is 40%
![val-loss_train-loss](https://user-images.githubusercontent.com/23450113/80319900-5e88c800-8813-11ea-9b53-0733c0b6e23f.png)

Training loss and Validation loss

![loss_train-val-Rnn](https://user-images.githubusercontent.com/23450113/80319905-60eb2200-8813-11ea-9ced-036e97a8a7cc.png)
