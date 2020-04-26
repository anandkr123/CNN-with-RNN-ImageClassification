# CNN-with-RNN-ImageClassification


1. COIL-100 dataset Challenge is a Kaggle Challenge. Image are rotated from 0-360 degress. 2,160
   images of 30 objects-72 poses per object. Image(128*128) is divided into 128 horizontal pixel rows
   that serves as input at continuous 128 time-stamps to a RNN network.
2. Later, the RNN cell is unrolled in time for prediction of every object.

                           LOSS, ACCURACY on TRAINING and VALIDATION SET and PEDICTIONS on TEST SET

ACCURACY ON TRAINING SET is AROUND 50% while ACCURACY ON VALIDATION SET is 40%

![loss_train_val-acc_train_val-predictions](https://user-images.githubusercontent.com/23450113/80320095-a825e280-8814-11ea-9708-0d4e0453c2b0.png)


Training loss and Validation loss

![loss_train-val-Rnn](https://user-images.githubusercontent.com/23450113/80319905-60eb2200-8813-11ea-9ced-036e97a8a7cc.png)
