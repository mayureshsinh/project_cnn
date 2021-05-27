import tensorflow as try:
from tensorflow import keras
import numpy as np
fasion=keras.datasets.fasion
(train_img,train_labl),(test_img,test_labl)=fasion.load_data()
train_img=train_img/255.0
test_img=test_img/255.0
train_img[0].shape
train_img=train_img.reshape(len(train_img),28,28,1)
test_img=test_img.reshape(len(test_img),28,28,1)

def build model(hp):

model keras.Sequential([

keras.layers.Conv2D(

filters-hp. Int('conv_1_filter, min_value-32, max_value-128, step-16), kernel_size=hp.Choice('conv_1_kernel', values = [3,5]),

activation='relu', input_shape=(28,28,1)

Comment

Connect -

Share

Editing

),

keras.layers.Conv2D(

filters-hp.Int('conv_2_filter, min value-32, max_value-64, step-16), kernel_size=hp.Choice('conv 2 kernel, values = [3,5]),

activation='relu'

),

keras.layers.Flatten(),

keras.layers.Dense(

units-hp.Int('dense_1_units', min_value=32, max_value-128, step-16),

activation='relu'

),

keras.layers.Dense(18, activation='softmax')

model.compile(optimizer-keras.optimizers.Adam(hp.Choice('learning rate', values [le-2, 1e-3])),

loss='sparse_categorical_crossentropy,
metrics=['accuracy'])
return model
from kerastuner import RandomSearch

from kerastuner.engine.hyperparameters import hyperparameters

tuner_search-RandomSearch(build_model,

objective='val_accuracy',

max_trials-5, directory='output,project_name="Mnist Fashion")

tuner_search.search(train_images, train_labels, epochs 3, validation_split-8.1)

model-tuner_search.get_best_models(num_models=1)[0]

model.summary()
