import keras
import tensorflow as tf
from keras.layers import Softmax, Dense, Reshape, Conv1D, Flatten, Conv2D
import neural_structured_learning as nsl


SAMPLE_RATE = 2000
NUM_CLASSES = 10


def conv_2d_model():
    model = tf.keras.Sequential([
        Conv2D(64, 3, activation='relu', input_shape=(128, 126, 1)),
        Conv2D(1, 3, 2, activation='relu'),

        Flatten(),
        Dense(16, activation='relu'),
        Dense(NUM_CLASSES),
        Softmax()
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model


def conv_1d_model():
    model = tf.keras.Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(SAMPLE_RATE, 1)),

        # dilated convolution block
        Conv1D(32, 5, dilation_rate=1, activation='relu'),
        Conv1D(32, 5, dilation_rate=2, activation='relu'),
        Conv1D(32, 5, dilation_rate=4, activation='relu'),
        Conv1D(32, 5, dilation_rate=8, activation='relu'),

        Conv1D(1, 3, 2, activation='relu'),

        Flatten(),
        Dense(16, activation='relu'),
        Dense(NUM_CLASSES),
        Softmax()
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model


def simple_mlp_model(adversarial=False):
    model = tf.keras.Sequential([
        Dense(16, activation='relu', input_shape=(SAMPLE_RATE,)),
        Dense(NUM_CLASSES),
        Softmax()
    ])

    # add adversarial regularization
    if adversarial:
        adv_config = nsl.configs.make_adv_reg_config(
            multiplier=0.25,
            adv_step_size=0.003,
            clip_value_min=-128,
            clip_value_max=127,
            pgd_epsilon=0.25,
            pgd_iterations=10
        )
        model = nsl.keras.AdversarialRegularization(model, ['label'], adv_config=adv_config)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model
