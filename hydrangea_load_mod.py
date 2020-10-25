import os

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# Initial Settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.chdir(r'D:\Udemy\CNN course\hydrangea')

# Loading Model
my_model = load_model(filepath='hydrangea_cnn_0.95.h5')
print(my_model.summary())

# Parameters: Weights and Biases
print('Hydrangea CNN last layer bias:')
print(my_model.get_weights()[-1])
print('Hydrangea CNN last layer weights:')
print(my_model.get_weights()[-2])