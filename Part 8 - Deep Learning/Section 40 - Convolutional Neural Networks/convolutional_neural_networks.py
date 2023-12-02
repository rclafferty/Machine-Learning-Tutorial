################################
# CONVOLUTIONAL NEURAL NETWORK #
################################

### PART 1: Data Preprocessing ###

# Importing the libraries #
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.models import Sequential
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# Part 1 - Data Preprocessing #

# Preprocessing the training set
train_datagen = ImageDataGenerator( # Feature scaling
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)
training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary'
)

# Preprocessing the test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary'
)

# Part 2 - Building the CNN #

# Initializing the CNN
cnn = Sequential()

# Step 1 - Convolution
cnn.add(Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = [64, 64, 3]))

# Step 2 - Pooling
cnn.add(MaxPool2D(strides = 2)) # pool_size = (2,2), padding = 'valid'

# Adding a second convolutional layer
cnn.add(Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
cnn.add(MaxPool2D(strides = 2)) # pool_size = (2,2), padding = 'valid'

# Step 3 - Flattening
cnn.add(Flatten())

# Step 4 - Full Connection
cnn.add(Dense(units = 128, activation = 'relu'))

# Step 5 - Output Layer
cnn.add(Dense(units = 1, activation = 'sigmoid'))

# Part 3 - Training the CNN #
TRAINING_EPOCHS = 25

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the training set and evaluating it on the test set
cnn.fit(x = training_set, validation_data = test_set, epochs = TRAINING_EPOCHS)

# Part 4 - Making a single prediction #
test_image_1 = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image_1 = image.img_to_array(test_image_1)
test_image_1 = np.expand_dims(test_image_1, axis = 0)
result_1 = cnn.predict(test_image_1)

prediction = 'Unknown'
# training_set.class_indices
if result_1[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
print(f'Prediction 1: {prediction}')

test_image_2 = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
test_image_2 = image.img_to_array(test_image_2)
test_image_2 = np.expand_dims(test_image_2, axis = 0)
result_2 = cnn.predict(test_image_2)

if result_2[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
print(f'Prediction 2: {prediction}')

test_image_3 = image.load_img('dataset/single_prediction/cat_or_dog_3.jpg', target_size = (64, 64))
test_image_3 = image.img_to_array(test_image_3)
test_image_3 = np.expand_dims(test_image_3, axis = 0)
result_3 = cnn.predict(test_image_3)

prediction = 'Unknown'
# training_set.class_indices
if result_3[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
print(f'Prediction 3: {prediction}')

test_image_4 = image.load_img('dataset/single_prediction/cat_or_dog_4.jpg', target_size = (64, 64))
test_image_4 = image.img_to_array(test_image_4)
test_image_4 = np.expand_dims(test_image_4, axis = 0)
result_4 = cnn.predict(test_image_4)

if result_4[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
print(f'Prediction 4: {prediction}')