# CONVOLUTIONAL NUERAL NETWORK (CNN)


# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html
# OR   pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# If something doesnt work try to ugrade pip if u using pip
# pip install --upgrade pip


# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__



# ---------------------------- Part 1 - Data Preprocessing -----------------------------------

# We have to manually preprocess the dataset by seperating different types of images and splitting it into test and train set.
# Down load datset from  https://www.superdatascience.com/pages/deep-learning  and then got to the CNN section.
# The datset is has to be divided into two parts training and testing dataset, and further divided into labels/categories like in thos case it is CAT and DOG. 


# Generating and Importing datset images

# Generating images for the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Generating images for the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)

# Creating the Training set
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Creating the Test set
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')




# --------------------------------- Part 2 - Building the CNN ------------------------------------



# Initialising the CNN
cnn = tf.keras.models.Sequential()


# Step 1 - Convolution
# we have to pass the number of filters/Feature detector,which is actually the number of feature maps formed aswell.
# we also have to pass the number of rows and column of filter of the filters/Feature detector using "kernel_size".
# we have to pass the border mode in "padding".
# we have to pass in input_shape that is the shape of the input image on which we will apply the Feature detector. 
# we also have to pass the activation function that we want to use.
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[64, 64, 3]))

# Step 2 - Pooling
# we need to pass the size "pool_size".
# we need to pass the size of stride.
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
# we need to pass the number of nodes/nuerons as "units" in the hidden layer.
# we also need to pass the number of activation function which we want to use.
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
# we need to pass the number of nodes/nuerons "units" in the output layer.
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))




# ----------------------------- Part 3 - Training the CNN ---------------------------------------



# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit_generator(training_set,
                  steps_per_epoch = 334,
                  epochs = 25,
                  validation_data = test_set,
                  validation_steps = 334)




