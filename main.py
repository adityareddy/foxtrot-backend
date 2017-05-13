import logging

import random
import subprocess
import numpy as np
import tensorflow as tf
from flask_cors import CORS
from flask import Flask
from flask import request
from flask import jsonify
from collections import deque
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

app = Flask(__name__)
CORS(app)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.9    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.e_decay = .99
        self.e_min = 0.05
        self.learning_rate = 0.01
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(20, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(20, activation='tanh', kernel_initializer='uniform'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=RMSprop(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

#    def act(self, state):
#        if np.random.rand() <= self.epsilon:
#            return random.randrange(self.action_size)
#        act_values = self.model.predict(state)
#        return np.argmax(act_values[0])  # returns action

    def act(self, state):
        act_values = self.model.predict(state)
        return act_values[0]

    def replay(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        X = np.zeros((batch_size, self.state_size))
        Y = np.zeros((batch_size, self.action_size))
        for i in range(batch_size):
            state, action, reward, next_state, done = minibatch[i]
            target = self.model.predict(state)[0]
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * \
                            np.amax(self.model.predict(next_state)[0])
            X[i], Y[i] = state, target
        self.model.fit(X, Y, batch_size=batch_size, nb_epoch=1, verbose=0)
        if self.epsilon > self.e_min:
            self.epsilon *= self.e_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

global graph
agent = DQNAgent(4, 4)
graph = tf.get_default_graph()

@app.route('/')
def hello():
    return 'Hello agent!'
    
@app.route('/generate')
def generate():
    brand_text = None
    state = request.args.get('state').split(',')
    state = np.array([float(item) for item in state])
    with graph.as_default():
      action = agent.act(state.reshape(1, 4)).tolist()
    try:
      brand_text = subprocess.check_output(["./text-to-svg", request.args.get('text'), "-s", request.args.get('size')])
    except subprocess.CalledProcessError as e:
      print(str(e))
    result = {
        'action': action,
        'brand_text': brand_text
    }
    return jsonify(result)
	
@app.route('/remember')
def remember():
    state = request.args.get('state').split(',')
    state = np.array([float(item) for item in state])
    next_state = request.args.get('next_state').split(',')
    next_state = np.array([float(item) for item in state])
    reward = float(request.args.get('reward'))
    done = request.args.get('done')
    agent.remember(state.reshape(1, 4), reward, next_state.reshape(1, 4), done)
    return 'remembered!'
    
@app.route('/replay')
def replay():
    agent.replay(4)
    return 'replay!'
    
@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=True)
