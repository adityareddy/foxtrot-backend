import logging

import random
import monogram
import subprocess
import numpy as np
import tensorflow as tf
from dqn import DQNAgent
from flask_cors import CORS
from flask import Flask
from flask import request
from flask import jsonify

app = Flask(__name__)
CORS(app)

global graph
agent = DQNAgent(4, 4)
graph = tf.get_default_graph()

@app.route('/')
def hello():
    return 'Hello agent!'

@app.route('/generate')
def generate():
    brand_path = None
    slogan_path = None
    brand_text = request.args.get('brand')
    slogan_text = request.args.get('slogan')
    
    state = request.args.get('state').split(',')
    state = np.array([float(item) for item in state])
    with graph.as_default():
      action = agent.act(state.reshape(1, 4)).tolist()

    try:
      brand_path = subprocess.check_output(
          [
              "./font_text/text-to-svg", 
              brand_text
            ])
      if slogan_text:
          slogan_path = subprocess.check_output(
              [
                  "./font_text/text-to-svg", 
                  slogan_text
                ])
    except subprocess.CalledProcessError as e:
      print(str(e))
    result = {
        'action': action,
        'brand_path': str(brand_path),
        'slogan_path': str(slogan_path),
        'monogram': random.choice([monogram.generate_monogram(brand_text[0].upper()), None]),
        'background_color': random.choice(['#171b1e','#BF2F2F'])
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
