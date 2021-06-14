import numpy as np
import pandas as pd
import requests
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
import gradio as gr
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import warnings
warnings.filterwarnings('ignore')
model_vgg16 = VGG16(weights='imagenet', include_top=False)
for layer in model_vgg16.layers:
    layer.trainable = False
img_width, img_height = 150, 150
train_data_dir = requests.get("https://drive.google.com/drive/folders/1IdT1sAVHQXzjoyLq7nPewRlOWqd-_r1k?usp=sharing")
val_data_dir = requests.get("https://drive.google.com/drive/folders/1LPdKb7Ru74MCzgOEEnrfUsFcL1Ts7e4e?usp=sharing")
model_weights_file = 'model_vgg16.h5'
nb_train_samples = 4
nb_val_samples = 4
nb_epochs = 5
put1 = Input(shape=(img_width, img_height, 3))
output_resnet = model_vgg16(put1)
x = Flatten()(output_resnet)
x = Dense(64, activation='relu')(x)
x = Dense(10, activation='softmax')(x)
model=Model(inputs=put1,outputs=x)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = tensorflow.keras.preprocessing.image_dataset_from_directory(train_data_dir, target_size=(img_width, img_height),batch_size=32, class_mode='categorical')
validation_generator = tensorflow.keras.preprocessing.image_dataset_from_directory(val_data_dir, target_size=(img_width, img_height),batch_size=32,class_mode='categorical')
history = model.fit( train_generator, epochs=2,steps_per_epoch=len(train_generator),validation_steps=len(validation_generator))
class_names=['beagle', 'chihuahua', 'doberman', 'french_bulldog', 'golden_retriever', 'malamute', 'pug', 'saint_bernard', 'scottish_deerhound', 'tibetan_mastiff']

def predict_image(img):
  img_4d=img.reshape(-1,150,150,3)
  prediction=model.predict(img_4d)[0]
  return {class_names[i]: float(prediction[i]) for i in range(10)}
image = gr.inputs.Image(shape=(150,150))
label = gr.outputs.Label(num_top_classes=10)
r=gr.Interface(fn=predict_image, inputs=image, outputs=label,interpretation='default').launch(debug='True')
r.launch()
