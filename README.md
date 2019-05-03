# **Traffic Sign Classifier**
<img src="examples/grayscale.jpg" alt="Combined Image" />

This project help to classify the Trafic Signs using deep learning model with tensorflow:
1. Conv1 = (Input) 32x32x1 --5x5--> Con1=32x32x32 --2x2--> Pool1=16x16x32
2. Conv2 = (1) 16x16x32 --5x5--> Con2=16x16x64 --2x2--> Pool2=8x8x64
3. Conv3 = (2) 8x8x64 --5x5--> Con3=8x8x128 --2x2--> Pool3=4x4x128
4. ---Conv1|Conv2|Conv3---> 16x16x32 + 8x8x64 + 4x4x128 = 14336
5. Size = 14336 | Dropout = 7168
6. Dense1 = 14336 --> 400
7. Dense2 = 400 --> 120
8. Dense3 = 120 --> 84
9. Dense4 = 84 --> 43 = P

Current validationi accuracy: ~95%

More details at `Traffic_Sign_Classifier.html`

You can download the data set at [this](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip).
