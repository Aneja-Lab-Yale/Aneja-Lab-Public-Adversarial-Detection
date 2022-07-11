# import dependencies
import tensorflow as tf
from tensorflow.keras.applications import VGG19, ResNet50, DenseNet121
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Convolution2D,BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



def resnet_model(input_shape):
    resnet_base = ResNet50(input_shape=input_shape, weights='imagenet', include_top=False)
    model = Sequential()
    model.add(resnet_base)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    # optimizer hyperparameters
    learning_rate =0.0002
    sgd = SGD(lr=learning_rate)
    model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=[tf.keras.metrics.BinaryAccuracy()])
    return model


def densenet_model(input_shape):
    model_d = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    x = model_d.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    preds = Dense(2, activation='softmax')(x)  # FC-layer
    model = Model(inputs=model_d.input, outputs=preds)
    learning_rate = 0.002
    sgd = SGD(lr=learning_rate)
    model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=[tf.keras.metrics.BinaryAccuracy()])
    return model
