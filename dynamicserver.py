from flask import Flask, render_template, jsonify

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)