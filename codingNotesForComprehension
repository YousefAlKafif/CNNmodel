DATASET STRUCTURE & Pre-Processing:-
	* 
Not a .csv data file.
	* 
Use keras.
	* 
We split the data ourselves by using folders.

		* 
Training:-

			* 
dataset\training_set\cats\cats1.jpg #up to 4000 so 4000 images
			* 
dataset\training_set\dogs\dogs1.jpg #up to 4000 so 4000 images
		* 
Test:-

			* 
dataset\test_set\cats\cats4000.jpg #up to 5000 so 1000 images
			* 
dataset\test_set\dogs\dogs4000.jpg #up to 5000 so 1000 images
	* 
We scale it using:-



from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


PART 1 -> BUILDING THE CNN
Importing the Keras libraries and packages & Initialization:-

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN

classifier = Sequential()
	* 
Declares 'classifier' as a sequence of layers.



Step 1 - CONVOLUTION LAYER :-

classifier.add(Conv2D(32, (3, 3), input_shape = (32, 64, 64, 3), activation = 'relu'))
	* 
First argument is filters -> 32 *You double the filters in every consecutive layer

		* 
The # of filters. -> Also our # of feature maps obviously.
	* 
Second argument is the (Height, Width) -> (3, 3)

		* 
Our (height, width) of our matrix. Here it is a 3x3 matrix.
	* 
input_shape argument is the (Batch Size, Rows/Height, Columns/Width, Channels) -> (32, 64, 64, 3)

		* 
Our input specifications of our images. i.e. what our conv layer should expect to receive.

			* 
NOTE: We have to resize the images during our importation of the image folders in the fitting stage.
		* 
(batch size, height, width, channels)

