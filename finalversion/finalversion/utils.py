from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.layers import LSTM, Dense, Flatten
import streamlit as st
import keras
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight


def load_model():
    num_classes = 3
    timesteps = 1
    input_shape = (timesteps, 2548)


    model = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=input_shape),
            keras.layers.Conv1D(filters=82, kernel_size=3, activation='relu', padding='causal'),
            LSTM(units = 256),
            keras.layers.Dense(num_classes, activation="softmax"),
        ])
    # model.load_weights("test.h5")
    return model