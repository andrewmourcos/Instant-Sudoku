# git subtree push --prefix webapp heroku master
# requirements.txt --> using old opencv for a reason
# static has to be "/tmp" for heroku

import os
from flask import Flask, render_template, request, after_this_request
from script import *
import cv2
import numpy as np
import joblib
from sklearn import datasets, neighbors, linear_model
from sklearn.externals import joblib
from sklearn.svm import LinearSVC

__author__ = 'andrewmourcos'

app = Flask(__name__, static_folder='tmp')
# app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'JPG'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
	return render_template("upload.html")

@app.route("/upload", methods=['POST'])
def upload():
	target = os.path.join(APP_ROOT, "tmp")
	print(target)
	if not os.path.isdir(target):
		os.mkdir(target)
	for file in request.files.getlist("file"):
		if allowed_file(file.filename):
			print(file)
			filename = file.filename
			destination = "/".join([target, filename])
			print(destination)
			file.save(destination)
			try:
				name, extracted_puzzle = solve(filename)
			except:
				print("no puzzle was found")
				try:
					pass
					# os.remove(destination)
				except Exception as error:
					app.logger.error("Error removing downloaded file")
				
				return render_template("upload.html", error_msg="No puzzle was detected, try a different angle")

			print(extracted_puzzle)
			# Clearing files
			# @after_this_request
			# def remove_file(response):
			# 	try:
			# 		os.remove(destination)
			# 		os.remove("/".join([target, name]))
			# 	except Exception as error:
			# 		app.logger.error("Error removing or closing downloaded file handle", error)
			# 	return response

			return render_template("upload.html", puzzle=filename, scan=name, recognized_puzzle=extracted_puzzle)
		else:
			print("Image format not supported")
			return render_template("upload.html", error_msg="This image format is unsupported")


if __name__ == "__main__":	
	app.run(port=4555, debug=True)