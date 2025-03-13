from flask import Flask, render_template, jsonify
from flask import request
import os

app = Flask(__name__)

@app.route('/')
def index():
    # This will serve the classifier.html template located in the templates folder.
    return render_template('classifier.html')

@app.route('/get_data')
def get_data():
    # Return dynamic JSON data.
    data = {"message": "Hello from Flask!", "value": 42}
    return jsonify(data)

@app.route('/upload', methods=['POST'])
def upload():
    # Check if the POST request contains the file part.
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in request.'})
    
    file = request.files['file']
    
    # Check if a file is selected.
    if file.filename == '':
        return jsonify({'message': 'No file selected.'})
    
    # Save the file in the uploads directory.
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    return jsonify({'message': 'File uploaded successfully.'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)