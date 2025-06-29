import os
from flask import Flask, render_template, jsonify, request, send_from_directory
import resnet18
import io
from PIL import Image
import argparse

app = Flask(__name__)

model_name = 'fullmix'
classes,trans=resnet18.ClassificationModel.load_classes(model_name)
grapes = resnet18.ClassificationModel(num_classes=len(classes))
grapes.load_model(model_name)


@app.route('/')
@app.route('/index')
def index():
    # Serve the classifier.html template
    return render_template('index.html')
@app.route('/contact')
def contact():
    # Serve the classifier.html template
    return render_template('contact.html')
@app.route('/articles')
def articles():
    # Serve the classifier.html template
    return render_template('articles.html')

@app.route('/about_us')
def about_us():
    # Serve the classifier.html template
    return render_template('about_us.html')

@app.route('/analyse')
def analyse():
    # Serve the classifier.html template
    return render_template('analyse.html')

@app.route('/home')
def home():
    # Serve the classifier.html template
    return render_template('home.html')


@app.route('/reaktor')
def reaktor():
    # Serve the classifier.html template
    return render_template('reaktor.html')



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
    try:
        # Read the file directly into memory and open as an image
        image_stream = io.BytesIO(file.read())
        image = Image.open(image_stream)
        
        # Optionally, if the model expects a different format, perform any necessary conversion here
        
        # Pass the image directly to the classifier.
        # You'll need to update classify_image in resnet18.ClassificationModel to work with a PIL Image.
        result_index,result_name,translation = grapes.classify_ram_image(image)

        
        return jsonify({'message': f'{translation}'})
    except Exception as e:
        return jsonify({'message': f'Error during classification: {str(e)}'})
# @app.route('/gallery')
# def gallery():
#     # Get list of filenames in the uploads folder.
#     image_folder = os.path.join(app.static_folder, 'galleryimages')
#     images = os.listdir(image_folder) if os.path.exists(image_folder) else []
#     return render_template('imagegallerie.html', images=images)

@app.route('/newpage')
def newpage():
    return render_template('newpage.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )
@app.route('/reaktor')
def production():
    return render_template('reaktor.html')

@app.route('/wuerze')
def wuerze():
    return render_template('wuerze.html')
@app.route('/thdreaktor')
def thdreaktor():
    return render_template('reaktorthd.html')

@app.route('/wahl')
def wahl():
    # Serve the classifier.html template
    return render_template('wahl.html')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Flask server")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Hostname to listen on')
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    app.run(debug=args.debug, host=args.host, port=args.port)