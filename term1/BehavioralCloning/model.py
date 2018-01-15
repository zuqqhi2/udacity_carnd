import csv
import cv2
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D, Activation, Convolution2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

### Get training and validation generator
def generator(samples, batch_size = 32, base_dir = './data', correction = 0.2):
  num_samples = len(samples)
  while 1: # Loop forever so the generator never terminates
    shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]

      images = []
      angles = []
      for batch_sample in batch_samples:
        # Load camera images
        for camera_id in range(3): # 0 is center, 1 is left, 2 is right
          fname = '%s/IMG/%s' % (base_dir, batch_sample[camera_id].split('/')[-1])
          img = cv2.imread(fname)
          images.append(img)

        # Load steering data
        angle = float(batch_sample[3])
        angles.append(angle)
        angles.append(angle + correction)
        angles.append(angle - correction)

        # Augmentation
        aug_images, aug_angles = [], []
        for img, angle in zip(images, angles):
          aug_images.append(img)
          aug_images.append(cv2.flip(img, 1))
          aug_angles.append(angle)
          aug_angles.append(angle * -1.0)
 
        # trim image to only see section with road
        X_train = np.array(aug_images)
        y_train = np.array(aug_angles)

        yield shuffle(X_train, y_train)

def get_train_generators(base_dir = './data'):
  # Load filename and steering data
  samples = []
  with open('%s/driving_log.csv' % base_dir) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
      samples.append(line)

  train_samples, validation_samples = train_test_split(samples, test_size = 0.2)

  train_generator = generator(train_samples)
  validation_generator = generator(validation_samples)

  return train_samples, train_generator, validation_samples, validation_generator


### Build network (NVIDIA Architecture)
def build_network():
  model = Sequential()
  model.add(Lambda(lambda x: x /127.5 - 1., input_shape = (160, 320, 3)))
  model.add(Cropping2D(cropping = ((70, 25), (0, 0))))
  model.add(Convolution2D(24, 5, 5, subsample = (2, 2), activation = 'relu'))
  model.add(Convolution2D(36, 5, 5, subsample = (2, 2), activation = 'relu'))
  model.add(Convolution2D(48, 5, 5, subsample = (2, 2), activation = 'relu'))
  model.add(Convolution2D(64, 3, 3, activation = 'relu'))
  model.add(Convolution2D(64, 3, 3, activation = 'relu'))
  model.add(Flatten())
  model.add(Dense(150))
  model.add(Dropout(0.2))
  model.add(Dense(90))
  model.add(Dropout(0.2))
  model.add(Dense(40))
  model.add(Dropout(0.2))
  model.add(Dense(1))
  model.compile(loss = 'mse', optimizer ='adam')

  return model


### Train network with drawing plot
def train(model, train_samples, train_generator, validation_samples, validation_generator):
  history_object = model.fit_generator(
    train_generator,
    samples_per_epoch = len(train_samples),
    validation_data = validation_generator,
    nb_val_samples = len(validation_samples), 
    nb_epoch = 5, verbose = 1)

  # Take a snapshot of the trained model
  model.save('model.h5')
  
  # plot the training and validation loss for each epoch
  plt.plot(history_object.history['loss'])
  plt.plot(history_object.history['val_loss'])
  plt.title('model mean squared error loss')
  plt.ylabel('mean squared error loss')
  plt.xlabel('epoch')
  plt.legend(['training set', 'validation set'], loc='upper right')
  plt.savefig('./train_history.png')


### Main logic
# Load training data
train_samples, train_generator, validation_samples, validation_generator = get_train_generators()

# Build network and Train the model
train(build_network(), train_samples, train_generator, validation_samples, validation_generator)
