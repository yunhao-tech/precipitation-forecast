from tensorflow import keras
from keras.models import Sequential 
from keras.layers import ConvLSTM2D, BatchNormalization, LeakyReLU
from keras.layers.convolutional import Conv3D


RE_HEIGHT = 200
RE_WIDTH = 200

def create_model():
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(7, 7),
                    input_shape=(18, RE_WIDTH, RE_HEIGHT,1),
                    padding='same', activation=LeakyReLU(alpha=0.01), 
                    return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=64, kernel_size=(5, 5),
                    padding='same', activation=LeakyReLU(alpha=0.01), 
                    return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                    padding='same', activation=LeakyReLU(alpha=0.01), 
                    return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=64, kernel_size=(1, 1),
                    padding='same', activation=LeakyReLU(alpha=0.01), 
                    return_sequences=True))
    model.add(BatchNormalization())
    model.add(Conv3D(filters=1, kernel_size=(3,3,3),
                activation='sigmoid',
                padding='same', data_format='channels_last'))
    return model

class ImageRegressor():
  def __init__(self):
    model = create_model()
    model.compile(loss='binary_crossentropy', optimizer='adadelta')
    self.epochs = 5
    self.batch_size = 1
    self.model = model
  
  def fit(self, X, y):
    self.model.fit(
        X,
        y,
        batch_size = self.batch_size,
        epochs = self.epochs,
        validation_data = None,
        verbose = 1,
    )

  def predict(self, X):
    return self.model.predict(X)

def get_estimator():
  model = ImageRegressor()
  return model