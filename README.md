# CNNmodel
This is a convolutional neural network model I created to classify cats and dogs. I used the Keras API and ran on the TensorFlow backend. I managed to achieve a final validation accuracy of 87.3 % and training accuracy of 95.25% , indicating that the model has slightly overfitted.


Some conventional methods I used that increased my accuracy :-
- Building a fully connected ANN ontop of the CNN layers. Which used 'Adam' as the gradient descent algorithm.
- Data augmentation to ensure a more robust model is trained, also because my dataset only consisted of 5,000 images of cats and 5,000 images of dogs.
- Multiple layers of Convolution + Max Pooling in order to reduce size whilst retaining valuable data - thus increasing computational speed and accounting for spatial variance.
- The 'relu' rectifier activation function to eliminate negative values, in order to decrease linearity.
- Doubling the # of feature detectors in consecutive convolutional layers.
- The 'dropout' method with a rate of 0.5 in the fully connected ANN layer to prevent overfitting.

Please view my 'CNNcomprehensionNotes' pdf if you'd like a step-by-step analysis of my work and how I did it.
View here : https://github.com/YousefAlKafif/CNNonKeras/blob/master/CNNcomprehensionNotes.pdf
