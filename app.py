import torch
from flask import Flask, request, render_template
from PIL import Image
import numpy as np
from io import BytesIO
import base64

# Load a YOLOv5 model (choose a variant, e.g., 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x')
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # Using a larger model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return "No file uploaded"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No file selected"
    
    if file and allowed_file(file.filename):
        img = Image.open(file.stream)
        
        # Convert image to RGB if it is in RGBA mode
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        results = model(img)
        
        # Save original image
        img_byte_array = BytesIO()
        img.save(img_byte_array, format='JPEG')
        img_byte_array = base64.b64encode(img_byte_array.getvalue()).decode('utf-8')

        # Save result image
        results_img = Image.fromarray(results.render()[0])
        if results_img.mode == 'RGBA':
            results_img = results_img.convert('RGB')
        results_img_byte_array = BytesIO()
        results_img.save(results_img_byte_array, format='JPEG')
        results_img_byte_array = base64.b64encode(results_img_byte_array.getvalue()).decode('utf-8')

        # Extract object names and count from YOLOv5 detection results
        predictions = results.xyxy[0].numpy()
        objects_detected = predictions[:, 5]  # Class indices
        objects_count = len(np.unique(objects_detected))
        object_names = [model.names[int(idx)] for idx in objects_detected]
        
        # Log object names for debugging
        print("Detected objects:", object_names)

        return render_template('result.html', 
                               original_img=img_byte_array, 
                               results_img=results_img_byte_array, 
                               object_names=object_names, 
                               objects_count=objects_count)
    else:
        return "Invalid file type"

@app.route('/camera')
def camera_page():
    return render_template('camera.html')

@app.route('/detect_from_camera', methods=['POST'])
def detect_from_camera():
    # Open the camera
    import cv2
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to PIL Image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Perform detection
        results = model(img)
        
        # Convert results image to JPEG format
        results_img = Image.fromarray(results.render()[0])
        if results_img.mode == 'RGBA':
            results_img = results_img.convert('RGB')
        results_img_byte_array = BytesIO()
        results_img.save(results_img_byte_array, format='JPEG')
        results_img_byte_array = base64.b64encode(results_img_byte_array.getvalue()).decode('utf-8')

        # Extract object names and count from YOLOv5 detection results
        predictions = results.xyxy[0].numpy()
        objects_detected = predictions[:, 5]  # Class indices
        objects_count = len(np.unique(objects_detected))
        object_names = [model.names[int(idx)] for idx in objects_detected]

        # Log object names for debugging
        print("Detected objects from camera:", object_names)

        # Close the camera
        cap.release()
        cv2.destroyAllWindows()

        return render_template('result.html', 
                               original_img=None,  # No original image for camera
                               results_img=results_img_byte_array, 
                               object_names=object_names, 
                               objects_count=objects_count)
    
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)
