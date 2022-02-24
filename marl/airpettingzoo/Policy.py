from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, Concatenate, concatenate, Input, Reshape
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
#from rl.core import MultiInputProcessor
from rl.processors import MultiInputProcessor

class Policy(object):

    def __init__(self, observation_size, state_size, action_size, policy_name, learning_rate, optimizer):

        self.observation_size = observation_size  # Input image with frame stacking size
        self.state_size = state_size  # Input state size (includes combined information regarding the neighbour drones)

        self.action_size = action_size
        self.weight_backup = policy_name
        self.learning_rate = learning_rate
        self.optimizer_model = optimizer

        self.model = self._build_model()
        self.model_target = self._build_model()

    def _build_model(self):
        # define two sets of inputs
        InputA = Input(shape=((self.observation_size,)))
        InputB = Input(shape=((self.state_size,)))

        # Convolutional Neural Network (CNN)
        x = Conv2D(32, 8, strides=4, activation="relu")(InputA)
        x = Conv2D(64, 4, strides=2, activation="relu")(x)
        x = Conv2D(64, 3, strides=1, activation="relu")(x)
        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        x = Model(inputs=InputA, outputs=x)

        # Multi-Layer Perception (MLP)
        y = Dense(64, activation="relu")(InputB)
        y = Dense(32, activation="relu")(y)
        y = Dense(16, activation="relu")(y)
        y = Model(inputs=InputB, outputs=y)

        # combine the output of the two branches
        combined = Concatenate(axis=-1)([x.output, y.output])

        # Adding few more FC layers
        z = Dense(256, activation="relu")(combined)
        action = Dense(self.action_size , activation="linear")(z)
           
        # Final mixed-data model
        model = Model(inputs=[x.input, y.input], outputs=action)

        # Select optimizers : Adam or RMSProp
        if self.optimizer_model == 'Adam':
            optimizer = keras.optimizers.Adam(lr=self.learning_rate, clipnorm=1.)
        elif self.optimizer_model == 'RMSProp':
            optimizer = keras.optimizers.RMSprop(lr=self.learning_rate, clipnorm=1.)
        else:
            print('Invalid optimizer!')

        model.compile(loss=keras.losses.Huber(), optimizer=optimizer)

        return model

    def train(self, x, y, sample_weight=None, epochs=1, verbose=0):  # x is the input to the network and y is the output

        self.model.fit(x, y, batch_size=len(x), sample_weight=sample_weight, epochs=epochs, verbose=verbose)

    def predict(self, state, target=False):
        if target:  # get prediction from target network
            return self.model_target.predict(state)
        else:  # get prediction from local network
            return self.model.predict(state)

    def predict_one_sample(self, state, target=False):
        return self.predict(state.reshape(1,[self.observation_size, self.state_size]), target=target).flatten()

    def update_target_model(self):
        self.model_target.set_weights(self.model.get_weights())

    def save_model(self):
        self.model.save(self.weight_backup)