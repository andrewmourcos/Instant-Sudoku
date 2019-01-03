import os
from flask import Flask, render_template, request
from script import *
import cv2
import numpy as np
import joblib
from sklearn import datasets, neighbors, linear_model
from sklearn.externals import joblib
from sklearn.svm import LinearSVC

__author__ = 'andrewmourcos'

# app = Flask(__name__, static_folder='images')
app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
	return render_template("upload.html")

@app.route("/upload", methods=['POST'])
def upload():
	target = os.path.join(APP_ROOT, "static")
	print(target)

	if not os.path.isdir(target):
		os.mkdir(target)

	for file in request.files.getlist("file"):
		print(file)
		filename = file.filename
		destination = "/".join([target, filename])
		print(destination)
		file.save(destination)
		name = solve(filename)
	return render_template("upload.html", puzzle=filename, scan=name)

if __name__ == "__main__":	
	app.run(port=4555, debug=True)