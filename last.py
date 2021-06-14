import numpy as np
import pandas as pd
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Flatten, Dense
from keras.callbacks import Callback, ModelCheckpoint
import gradio as gr
from keras.preprocessing import image

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,GlobalAveragePooling2D,Dropout
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import warnings
warnings.filterwarnings('ignore')
model_vgg16 = VGG16(weights='imagenet', include_top=False)
for layer in model_vgg16.layers:
    layer.trainable = False
img_width, img_height = 150, 150
train_data_dir = r'C:\Users\viji\Documents\train_new'
val_data_dir = r'C:\Users\viji\Desktop\test_new'
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
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,\
     Dropout,Flatten,Dense,Activation,\
     BatchNormalization
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height),
                                                    batch_size=32, class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(val_data_dir, target_size=(img_width, img_height),
                                                        batch_size=32,class_mode='categorical')
class_name=[a for a in os.listdir('/content/drive/MyDrive/train_new')]
class_names=sorted(class_name)
print(class_names)

def predict_image(img):
  img_4d=img.reshape(-1,150,150,3)
  prediction=model.predict(img_4d)[0]
  return {class_names[i]: float(prediction[i]) for i in range(10)}
image = gr.inputs.Image(shape=(150,150))
label = gr.outputs.Label(num_top_classes=10)
r=gr.Interface(fn=predict_image, inputs=image, outputs=label,interpretation='default').launch(debug='True')
r.launch()