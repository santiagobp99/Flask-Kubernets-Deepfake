import numpy as np
import os
import matplotlib.pyplot as plt
import json
import time
import _pickle as pickle
from tqdm import tqdm
from PIL import Image
import pandas as pd
import random
from IPython.display import Video
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
import sys
import statistics
import pickle
from statistics import mode

from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input as pre

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer
from sklearn.preprocessing import OneHotEncoder

from pathlib import Path

import tensorflow as tf
from keras.layers import Input
import keras
from mtcnn.mtcnn import MTCNN
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.models import load_model
from flask import Flask, request, jsonify, url_for, render_template, flash, redirect
import uuid
import os
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image, ImageFile
from io import BytesIO
from keras.preprocessing import image
import sys, os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LABELS = ['FAKE', 'REAL']
IMG_SIZE = 299
    
ALLOWED_EXTENSION = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp4'])
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3

# Metodos previos

def build_feature_extractor():
    feature_extractor = InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


feature_extractor = build_feature_extractor()


# Flask

def allowed_file(filename):
    return '.' in filename and \
     filename.rsplit('.',1)[1] in ALLOWED_EXTENSION

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/')
def upload_form():
    return render_template('upload.html', prediction={})
    
    
def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.route('/shutdown')
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

@app.route('/display/<filename>')
def display_video(filename):
    #print('display_video filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_video filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app.
    app.run(host='0.0.0.0', port=8080, debug=True)