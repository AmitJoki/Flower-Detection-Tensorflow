from flask import Flask, render_template, request, jsonify
from werkzeug import secure_filename
from retrain import predict

app = Flask(__name__)

@app.route('/')
def hello_world():
    return '<h1>Hello, World!</h1><a href="/upload">Start detecting the flowers!</a>'

@app.route('/upload')
def upload():
   return render_template('predict.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      print(request.files)
      f = request.files['file']
      f.save('target')

      return jsonify(predict('target'))