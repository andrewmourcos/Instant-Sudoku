from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug import secure_filename
import os

app = Flask(__name__)

IMAGE_FOLDER = 'image'
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

# map return code to a URL
@app.route('/')
def index():
   return render_template('/index.html', image = "../image/yeet.png")

@app.route('/', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['file']
		f.save("image/%s" %secure_filename("yeet.png"))

# only run if this is main file
if __name__ == "__main__":
	app.run(debug=True)