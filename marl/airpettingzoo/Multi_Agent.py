import numpy as np
import random

from Policy import Policy

MAX_EPSILON = 1.0
MIN_EPSILON = 0.01

class Agent(object):
    
    epsilon = MAX_EPSILON

    def __init__(self, observation_size, state_size, action_size, policy_name, arguments):
        self.state_size = state_size
        self.observation_size = observation_size
        self.action_size = action_size
        self.learning_rate = arguments['learning_rate']
        self.optimizer = arguments['optimizer']
        self.gamma = 0.95
        self.policy = Policy(self.observation_size, self.state_size, self.action_size, policy_name, self.learning_rate, self.optimizer)

        self.update_target_frequency = arguments['target_frequency']
        self.max_exploration_step = arguments['maximum_exploration']
        self.batch_size = arguments['batch_size']
        self.step = 0

    def greedy_actor(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.policy.predict_one_sample(state))

    def decay_epsilon(self):
        # slowly decrease Epsilon based on our experience
        self.step += 1

        if self.test:
            self.epsilon = MIN_EPSILON
        else:
            if self.step < self.max_exploration_step:
                self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * (self.max_exploration_step - self.step)/self.max_exploration_step
            else:
                self.epsilon = MIN_EPSILON

    def update_target_model(self):
        if self.step % self.update_target_frequency == 0:
            self.policy.update_target_model()

    def replay(self):
        pass