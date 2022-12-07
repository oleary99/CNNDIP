import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL.Image as im
import os

import csv

def setup_threads():
  num_threads = 12
  os.environ["OMP_NUM_THREADS"] = "12"
  os.environ["TF_NUM_INTRAOP_THREADS"] = "12"
  os.environ["TF_NUM_INTEROP_THREADS"] = "12"

  tf.config.threading.set_inter_op_parallelism_threads(
      num_threads
  )
  tf.config.threading.set_intra_op_parallelism_threads(
      num_threads
  )
  tf.config.set_soft_device_placement(True)
def setup_training():
  base_dir = '/afhq'
  train_dir = "..\\..\\afhq\\train"
  training_set = tf.keras.preprocessing.image_dataset_from_directory(
  train_dir,
  seed=101,
  image_size=(200, 200),
  batch_size=32)
  class_names = training_set.class_names
  return training_set
def setup_validation():
  test_dir = "..\\..\\afhq\\val"
  validation_set = tf.keras.preprocessing.image_dataset_from_directory(
  test_dir,
  seed=101,
  image_size=(200, 200),
  batch_size=32)
  return validation_set
def setup_layer():
  data_augmentation = tf.keras.Sequential(
    [
      tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(200, 200,3)),
      #tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
      #tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
    ]
  )

def relu_model():
  return tf.keras.Sequential(
      [
      #data_augmentation,
      tf.keras.layers.experimental.preprocessing.Rescaling(1/255),
      tf.keras.layers.Conv2D(32, (3,3), padding='same', activation="relu",input_shape=(32, 32, 3)),
      #tf.keras.layers.Conv2D(32, (3,3), padding='same', input_shape=(32, 32, 3)),
      tf.keras.layers.MaxPooling2D((2, 2), strides=2),

      #tf.keras.layers.Conv2D(64, (3,3), padding='same', activation="relu"),
      #tf.keras.layers.Conv2D(64, (3,3), padding='same'),
      #tf.keras.layers.MaxPooling2D((2, 2), strides=2),

      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(50, activation="relu"),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(4, activation='softmax')
  ]
  )

def sigmoid_model():
  return tf.keras.Sequential(
      [
      #data_augmentation,
      tf.keras.layers.experimental.preprocessing.Rescaling(1/255),
      tf.keras.layers.Conv2D(32, (3,3), padding='same', activation="relu",input_shape=(32, 32, 3)),
      #tf.keras.layers.Conv2D(32, (3,3), padding='same', input_shape=(32, 32, 3)),
      tf.keras.layers.MaxPooling2D((2, 2), strides=2),

      #tf.keras.layers.Conv2D(64, (3,3), padding='same', activation="relu"),
      #tf.keras.layers.Conv2D(64, (3,3), padding='same'),
      #tf.keras.layers.MaxPooling2D((2, 2), strides=2),

      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(50, activation="sigmoid"),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(4, activation='softmax')
  ]
  )
def selu_model():
  return tf.keras.Sequential(
      [
      #data_augmentation,
      tf.keras.layers.experimental.preprocessing.Rescaling(1/255),
      tf.keras.layers.Conv2D(32, (3,3), padding='same', activation="selu",input_shape=(32, 32, 3)),
      #tf.keras.layers.Conv2D(32, (3,3), padding='same', input_shape=(32, 32, 3)),
      tf.keras.layers.MaxPooling2D((2, 2), strides=2),

      #tf.keras.layers.Conv2D(64, (3,3), padding='same', activation="relu"),
      #tf.keras.layers.Conv2D(64, (3,3), padding='same'),
      #tf.keras.layers.MaxPooling2D((2, 2), strides=2),

      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(50, activation="selu"),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(4, activation='softmax')
  ]
  )
def gelu_model():
  return tf.keras.Sequential(
      [
      #data_augmentation,
      tf.keras.layers.experimental.preprocessing.Rescaling(1/255),
      tf.keras.layers.Conv2D(32, (3,3), padding='same', activation="gelu",input_shape=(32, 32, 3)),
      #tf.keras.layers.Conv2D(32, (3,3), padding='same', input_shape=(32, 32, 3)),
      tf.keras.layers.MaxPooling2D((2, 2), strides=2),

      #tf.keras.layers.Conv2D(64, (3,3), padding='same', activation="relu"),
      #tf.keras.layers.Conv2D(64, (3,3), padding='same'),
      #tf.keras.layers.MaxPooling2D((2, 2), strides=2),

      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(50, activation="gelu"),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(4, activation='softmax')
  ]
  )
def tanh_model():
  return tf.keras.Sequential(
      [
      #data_augmentation,
      tf.keras.layers.experimental.preprocessing.Rescaling(1/255),
      tf.keras.layers.Conv2D(32, (3,3), padding='same', activation="tanh",input_shape=(32, 32, 3)),
      #tf.keras.layers.Conv2D(32, (3,3), padding='same', input_shape=(32, 32, 3)),
      tf.keras.layers.MaxPooling2D((2, 2), strides=2),

      #tf.keras.layers.Conv2D(64, (3,3), padding='same', activation="relu"),
      #tf.keras.layers.Conv2D(64, (3,3), padding='same'),
      #tf.keras.layers.MaxPooling2D((2, 2), strides=2),

      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(50, activation="tanh"),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(4, activation='softmax')
  ]
  )
def linear_model():
  return tf.keras.Sequential(
      [
      #data_augmentation,
      tf.keras.layers.experimental.preprocessing.Rescaling(1/255),
      tf.keras.layers.Conv2D(32, (3,3), padding='same', activation="linear",input_shape=(32, 32, 3)),
      #tf.keras.layers.Conv2D(32, (3,3), padding='same', input_shape=(32, 32, 3)),
      tf.keras.layers.MaxPooling2D((2, 2), strides=2),

      #tf.keras.layers.Conv2D(64, (3,3), padding='same', activation="relu"),
      #tf.keras.layers.Conv2D(64, (3,3), padding='same'),
      #tf.keras.layers.MaxPooling2D((2, 2), strides=2),

      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(50, activation="linear"),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(4, activation='softmax')
  ]
  )
def hard_sigmoid_model():
  return tf.keras.Sequential(
      [
      #data_augmentation,
      tf.keras.layers.experimental.preprocessing.Rescaling(1/255),
      tf.keras.layers.Conv2D(32, (3,3), padding='same', activation="hard_sigmoid",input_shape=(32, 32, 3)),
      #tf.keras.layers.Conv2D(32, (3,3), padding='same', input_shape=(32, 32, 3)),
      tf.keras.layers.MaxPooling2D((2, 2), strides=2),

      #tf.keras.layers.Conv2D(64, (3,3), padding='same', activation="relu"),
      #tf.keras.layers.Conv2D(64, (3,3), padding='same'),
      #tf.keras.layers.MaxPooling2D((2, 2), strides=2),

      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(50, activation="hard_sigmoid"),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(4, activation='softmax')
  ]
  )
def _model():
  return tf.keras.Sequential(
      [
      #data_augmentation,
      tf.keras.layers.experimental.preprocessing.Rescaling(1/255),
      tf.keras.layers.Conv2D(32, (3,3), padding='same', activation="relu",input_shape=(32, 32, 3)),
      #tf.keras.layers.Conv2D(32, (3,3), padding='same', input_shape=(32, 32, 3)),
      tf.keras.layers.MaxPooling2D((2, 2), strides=2),

      #tf.keras.layers.Conv2D(64, (3,3), padding='same', activation="relu"),
      #tf.keras.layers.Conv2D(64, (3,3), padding='same'),
      #tf.keras.layers.MaxPooling2D((2, 2), strides=2),

      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(50, activation="selu"),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(4, activation='softmax')
  ]
  )

def compile_model(model,training_set,validation_set,epochs):
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  return model.fit(training_set,validation_data = validation_set, epochs=epochs)
def graph_history(history):
  training_loss = history.history['loss']
  test_loss = history.history['val_loss']
  epoch_count = range(1, len(training_loss) + 1)
  plt.plot(epoch_count, training_loss, 'r--')
  plt.plot(epoch_count, test_loss, 'b-')
  plt.legend(['Training Loss', 'Test Loss'])
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.show();
def collect_data(history):
  f = open('Models.csv','a')
  writer = csv.writer(f)
  print(history.history)
  writer.writerow(['Loss']+ history.history['loss'])
  writer.writerow(['Accuracy']+ history.history['sparse_categorical_accuracy'])
  writer.writerow(['Val_Loss']+ history.history['val_loss'])
  writer.writerow(['Val_Accuracy']+ history.history['val_sparse_categorical_accuracy'])
  f.close()
  print(history.history)
def setup_csv():
  f = open('Models.csv','w')
  writer = csv.writer(f)
  writer.writerow(['Loss','Accuracy','Validation Loss','Validation Accuracy'])
  f.close()
def title_csv(Title):
  f= open('Models.csv', 'a')
  writer = csv.writer(f)
  writer.writerow([Title])
def run_model(activation,title):
  setup_threads()
  training_set = setup_training()
  validation_set = setup_validation()
  setup_layer()
  #model = relu_model()
  model = activation()
  history = compile_model(model,training_set,validation_set,2)
  title_csv(title)
  collect_data(history)
  #graph_history(history)

def main():
  setup_csv()
  run_model(sigmoid_model,'Sigmoid')
  run_model(relu_model,'Relu')
  run_model(selu_model,'Selu')
  run_model(gelu_model,'Gelu')
  run_model(tanh_model,'Tanh')
  run_model(linear_model,'Linear')
  run_model(hard_sigmoid_model,'Hard_Sigmoid')
  
main()