import zipfile
from flask import Blueprint, request, render_template, jsonify, url_for
from .model import load_model
from .utils import preprocess_image, inference, visualize_results, preprocess_images, visualize_prediction
from monai.transforms import Compose, Activations, AsDiscrete
import io
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

main = Blueprint('main', __name__)

# Load the model
model = load_model()

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
def upload_file():
    logger.debug("Received a POST request to /upload")
    
    # Process the uploaded file and perform inference
    if 'file' not in request.files:
        logger.error("No file uploaded")
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        logger.error("No file selected")
        return "No file selected", 400

    logger.info("File received: %s", file.filename)
    
    # Preprocess and run inference
    try:
        input_tensor = preprocess_image(file)
        logger.debug("Image preprocessing completed")
        
        output_tensor = inference(input_tensor)
        logger.debug("Inference completed")
        
        # Visualize and save results
        image_filename = visualize_results(input_tensor, output_tensor)
        logger.info("Results visualized and saved as %s", image_filename)
        
    except Exception as e:
        logger.exception("An error occurred during processing")
        return "Internal server error", 500

    # Pass the filename to the template
    return render_template('result.html', result_image=image_filename)


@main.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    zip_file = request.files['file']
    if not zip_file or not zip_file.filename.endswith('.zip'):
        return jsonify({'error': 'Invalid file format. Please upload a ZIP file.'}), 400

    try:
        # Read the ZIP file into memory
        zip_bytes = io.BytesIO(zip_file.read())
        with zipfile.ZipFile(zip_bytes, 'r') as zip_ref:
            # Preprocess the images from the zip archive
            image_tensor = preprocess_images(zip_ref)  # Pass the zip_ref to preprocess_images

            # Perform prediction
            with torch.no_grad():
                output = inference(image_tensor)
                output = output.cpu().numpy()
                response = {'prediction': output.tolist()}

            # Call visualization function
            image_filename = visualize_prediction(image_tensor.cpu().numpy(), output)

            return render_template('result.html', result_image=image_filename)

            # return jsonify(response)
    except Exception as e:
        raise e
        # return jsonify({'error': str(e)}), 500
