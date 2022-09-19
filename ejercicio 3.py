import tensorflow as tf
import urllib
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Model

# Crea un callback para detener el entrenamiento cuando llegue al 95%
# Your code here
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.95):
            print("Reached 95% accuracy so cancelling training!")
            self.model.stop_training = True

def get_data():
    _TRAIN_URL = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
    _TEST_URL = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"
    urllib.request.urlretrieve(_TRAIN_URL, 'horse-or-human.zip')
    local_zip = 'horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('data/horse-or-human/')
    zip_ref.close()
    urllib.request.urlretrieve(_TEST_URL, 'testdata.zip')
    local_zip = 'testdata.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('data/testdata/')
    zip_ref.close()

get_data()

def solution_model():
    train_dir = 'data/horse-or-human'
    validation_dir = 'data/testdata'

    print('total training horses images :', len(os.listdir(os.path.join(train_dir, 'horses'))))
    print('total training humans images :', len(os.listdir(os.path.join(train_dir, 'humans'))))
    print('total validation horses images :', len(os.listdir(os.path.join(validation_dir, 'horses'))))
    print('total validation humans images :', len(os.listdir(os.path.join(validation_dir, 'humans'))))

    # Carga los datos mediante ImageDataGenerator
    # Your code here

    # All images will be rescaled by 1./255.
    train_datagen = ImageDataGenerator(rescale=1.0/255.)
    test_datagen = ImageDataGenerator(rescale=1.0/255.)

    # Flow training images in batches using train_datagen generator

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=51,
                                                        class_mode='binary',
                                                        target_size=(300, 300))

    # Flow validation images in batches using train_datagen generator

    validation_generator = test_datagen.flow_from_directory(train_dir,
                                                            batch_size=25,
                                                            class_mode='binary',
                                                            target_size=(300, 300))

    model = tf.keras.models.Sequential([
        # Note the input shape specified on your first layer must be (300,300,3)
        # input_shape siempre debe ser igual a target_size
        # Your Code here
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(learning_rate=0.0001),
                  metrics=['accuracy'])

    callbacks = myCallback()
    model.fit(train_generator,
              epochs=10,
              steps_per_epoch=20,
              validation_data=validation_generator,
              verbose=1,
              validation_steps=10,
              callbacks=[callbacks])

    return model

if __name__ == '__main__':
    model = solution_model()