from flask import Flask, request, jsonify, send_file, abort, render_template
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from io import BytesIO
import uuid

app = Flask(__name__)

# Define your font path and default image path
font_path = "RunningYolo/fonts/AbingdonBold.otf"
default_image_path_back = "RunningYolo/Images/back.jpg"
default_image_path_front = "RunningYolo/Images/front.jpg"  # Specify the default image path
default_image_path_twice = "RunningYolo/Images/maillots-twice.jpg"

base_image_dir = "RunningYolo/Sites"


@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/create_site', methods=['POST'])
def create_site():
    new_site = request.form.get('new_site')
    site_path = os.path.join(base_image_dir, new_site)

    try:
        # Create the main site folder
        os.makedirs(site_path, exist_ok=True)
        
        # Create the domicile and exterieur subfolders
        os.makedirs(os.path.join(site_path, 'domicile'), exist_ok=True)
        os.makedirs(os.path.join(site_path, 'exterieur'), exist_ok=True)
        return "Site created successfully", 200
    except Exception as e:
        return f"Error creating site: {str(e)}", 500
    
@app.route('/get_sites', methods=['GET'])
def get_sites():
    # Get all directories (sites) in the base image directory
    sites = []
    for d in os.listdir(base_image_dir):
        site_path = os.path.join(base_image_dir, d)
        if os.path.isdir(site_path):
            sites.append({
                'name': d,
                'domicile': os.path.join(d, 'domicile'),
                'exterieur': os.path.join(d, 'exterieur')
            })
    return jsonify(sites)



@app.route('/upload', methods=['POST'])
def upload_file():
    files = request.files
    uploaded_files = []
    selected_folder = request.form.get('site') # Get the selected site from the form
    
    # Create the site folder path
    site_name, folder_type = selected_folder.split('/')
    site_folder = os.path.join(base_image_dir, site_name, folder_type)

    # Save uploaded files with unique names to the temporary directory
    if 'back' in files:
        file = files['back']
        if file.filename != '':
            back_filename = f"back.jpg"  # Unique filename
            back_path = os.path.join(site_folder, back_filename)
            file.save(back_path)
            uploaded_files.append(back_path)

    if 'front' in files:
        file = files['front']
        if file.filename != '':
            front_filename = f"front.jpg"  # Unique filename
            front_path = os.path.join(site_folder, front_filename)
            file.save(front_path)
            uploaded_files.append(front_path)

    if 'twice' in files:
        file = files['twice']
        if file.filename != '':
            twice_filename = f"twice.jpg"  # Unique filename
            twice_path = os.path.join(site_folder, twice_filename)
            file.save(twice_path)
            uploaded_files.append(twice_path)

    # Update default image paths to the newly uploaded files
    global default_image_path_back, default_image_path_front, default_image_path_twice
    if len(uploaded_files) > 0:
        default_image_path_back = uploaded_files[0] if 'back' in files else default_image_path_back
        default_image_path_front = uploaded_files[1] if 'front' in files else default_image_path_front
        default_image_path_twice = uploaded_files[2] if 'twice' in files else default_image_path_twice

    return "Files uploaded successfully", 200


def process_image_front(image, text1, side):

    # Resize the image
    scale = 60  # scale in percentage
    newWidth = int(image.shape[1] * scale / 100)
    newHeight = int(image.shape[0] * scale / 100)
    newDimension = (newWidth, newHeight)
    resizedImage = cv2.resize(image, newDimension, interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    grayImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur and thresholding
    blurredImage = cv2.GaussianBlur(grayImage, (5, 5), 0)
    _, thresholdImage = cv2.threshold(blurredImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresholdImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Make a copy of the resized image
    finalImage = np.copy(resizedImage)

    posiY1 = 70
    ccolor = (255, 255, 255)
    if (side == "exterieur"):
        ccolor = (64, 64, 64)
        posiY1 = posiY1 - 150

    # Get the second largest contour and calculate the centroid
    if len(sorted_contours) > 1:
        second_largest_contour = sorted_contours[1]
        M = cv2.moments(second_largest_contour)
        if M["m00"] != 0:  # To avoid division by zero
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Define text to display
            text = text1

            def add_text_with_pillow(image, text, position, font_path, font_size):
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)
                font = ImageFont.truetype(font_path, font_size)
                try:
                    # For newer versions of Pillow
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                except AttributeError:
                    # For older versions of Pillow
                    text_width, text_height = draw.textsize(text, font=font)

                # Center the text
                centered_position = (position[0] - text_width // 2, position[1] - text_height // 2)

                # Add text to the image
                draw.text(centered_position, text, font=font, fill=ccolor)
                return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            finalImage = add_text_with_pillow(finalImage, text, (cX, cY - posiY1), font_path, 45)

    return finalImage

def process_image(image, text1, text2, side):
    # Resize image
    scale = 60
    newWidth = int(image.shape[1] * scale / 100)
    newHeight = int(image.shape[0] * scale / 100)
    newDimension = (newWidth, newHeight)
    resizedImage = cv2.resize(image, newDimension, interpolation=cv2.INTER_AREA)

    # Convert image to grayscale
    grayImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurredImage = cv2.GaussianBlur(grayImage, (5, 5), 0)

    # Thresholding
    _, thresholdImage = cv2.threshold(blurredImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresholdImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Make a copy for final image
    finalImage = np.copy(resizedImage)
    posiY1 = 110
    posiY2 = 60
    ccolor = (255, 255, 255)
    if (side == "exterieur"):
        ccolor = (64, 64, 64)
        posiY1 = posiY1 + 100
        posiY2 = posiY2 + 100

    if len(sorted_contours) > 1:
        second_largest_contour = sorted_contours[1]
        M = cv2.moments(second_largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            def add_text_with_pillow(image, text, position, font_path, font_size):
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)
                font = ImageFont.truetype(font_path, font_size)

                try:
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                except AttributeError:
                    text_width, text_height = draw.textsize(text, font=font)

                centered_position = (position[0] - text_width // 2, position[1] - text_height // 2)
                draw.text(centered_position, text, font=font, fill=ccolor)
                return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            finalImage = add_text_with_pillow(finalImage, text1, (cX, cY - posiY1), font_path, 25)
            finalImage = add_text_with_pillow(finalImage, text2, (cX, cY - posiY2), font_path, 100)

    return finalImage


def process_image_twice(image, text1, text2, side):
    # Resize image
    scale = 60
    newWidth = int(image.shape[1] * scale / 100)
    newHeight = int(image.shape[0] * scale / 100)
    newDimension = (newWidth, newHeight)
    resizedImage = cv2.resize(image, newDimension, interpolation=cv2.INTER_AREA)

    # Convert image to grayscale
    grayImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurredImage = cv2.GaussianBlur(grayImage, (5, 5), 0)

    # Thresholding
    _, thresholdImage = cv2.threshold(blurredImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresholdImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Make a copy for final image
    finalImage = np.copy(resizedImage)
    posiY1 = 80
    posiY2 = 40
    posiY3 = 50
    posiX1 = 78
    posiX2 = 78
    posiX3 = 85
    ccolor = (255, 255, 255)

    if (side == "exterieur"):
        ccolor = (64, 64, 64)
        posiX1 = 0
        posiX2 = 0
        posiX3 = 85 + 85
        posiY1 = posiY1 + 70
        posiY2 = posiY2 + 70
        posiY3 = posiY3 + 50


    if len(sorted_contours) > 1:
        second_largest_contour = sorted_contours[1]
        M = cv2.moments(second_largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            def add_text_with_pillow(image, text, position, font_path, font_size):
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)
                font = ImageFont.truetype(font_path, font_size)

                try:
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                except AttributeError:
                    text_width, text_height = draw.textsize(text, font=font)

                centered_position = (position[0] - text_width // 2, position[1] - text_height // 2)
                draw.text(centered_position, text, font=font, fill=ccolor)
                return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            finalImage = add_text_with_pillow(finalImage, text1, (cX - posiX1, cY - posiY1), font_path, 25)
            finalImage = add_text_with_pillow(finalImage, text2, (cX - posiX2, cY - posiY2), font_path, 100)
            finalImage = add_text_with_pillow(finalImage, text2, (cX + posiX3, cY - posiY3), font_path, 25)

    return finalImage

@app.route('/process/back', methods=['POST'])
def process():
    # Get text1 and text2 from the request
    site_name = request.form.get('site')  # Expecting format like "cameroun/domicile"
    text1 = request.form.get('nom')
    text2 = request.form.get('numero')
    side = site_name.split("/")[1]

    # Check if both text1 and text2 are provided
    if not text1 or not text2 or not site_name:
        return abort(404, description="nom numero et site requis")
    
    # Construct the path for the selected site folder
    site_folder = os.path.join(base_image_dir, site_name)
    # Check if the folder exists
    if os.path.exists(site_folder):
        # Look for files starting with 'back'
        back_files = [f for f in os.listdir(site_folder) if f.startswith('back')]
        if not back_files:
            return "No file starting with 'back' found in the specified folder", 404

        # Use the first file found starting with 'back'
        back_image_path = os.path.join(site_folder, back_files[0])
        print("back_image_path", back_image_path)
        image = cv2.imread(back_image_path)
    else:
        return "Site folder not found", 404

    # if 'file' not in request.files:
    #     # Load and process the default image if no file is provided
    #     if os.path.exists(default_image_path_back):
    #         image = cv2.imread(default_image_path_back)
    #     else:
    #         return "Default image not found", 404
    # else:
    #     file = request.files['file']
    #     if file.filename == '':
    #         return "No selected file", 400

    #     # Read the image file
    #     in_memory_file = BytesIO()
    #     file.save(in_memory_file)
    #     in_memory_file.seek(0)
    #     image = Image.open(in_memory_file)
    #     image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Process the image
    final_image = process_image(image, text1, text2, side)

    # Convert the final image to bytes
    _, buffer = cv2.imencode('.png', final_image)
    final_image_bytes = BytesIO(buffer)

    return send_file(final_image_bytes, mimetype='image/png', as_attachment=True, download_name='processed_image.png')


@app.route('/process/front', methods=['POST'])
def process_and_send_image_front():
    site_name = request.form.get('site')  # Expecting format like "cameroun/domicile"
    text1 = request.form.get('numero')
    side = site_name.split("/")[1]

    if not text1 or not site_name:
        return abort(404, description="numero et site requis")
    
    # Construct the path for the selected site folder
    site_folder = os.path.join(base_image_dir, site_name)
    # Check if the folder exists
    if os.path.exists(site_folder):
        # Look for files starting with 'back'
        front_files = [f for f in os.listdir(site_folder) if f.startswith('front')]
        if not front_files:
            return "No file starting with 'back' found in the specified folder", 404

        # Use the first file found starting with 'back'
        front_image_path = os.path.join(site_folder, front_files[0])
        image = cv2.imread(front_image_path)
    else:
        return "Site folder not found", 404
    

    # if 'file' not in request.files:
    #     # Load and process the default image if no file is provided
    #     if os.path.exists(default_image_path_front):
    #         image = cv2.imread(default_image_path_front)
    #     else:
    #         return "Default image not found", 404
    # else:
    #     file = request.files['file']
    #     if file.filename == '':
    #         return "No selected file", 400
    #     # Read the image file
    #     in_memory_file = BytesIO()
    #     file.save(in_memory_file)
    #     in_memory_file.seek(0)
    #     image = Image.open(in_memory_file)
    #     image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
    processed_image_path = process_image_front(image, text1, side)
    # Convert the final image to bytes
    _, buffer = cv2.imencode('.png', processed_image_path)
    final_image_bytes = BytesIO(buffer)
    return send_file(final_image_bytes, mimetype='image/png',  as_attachment=True, download_name='processed_image_front.png')

@app.route('/process/twice', methods=['POST'])
def process_twice():
    # Get text1 and text2 from the request
    site_name = request.form.get('site')  # Expecting format like "cameroun/domicile"
    text1 = request.form.get('nom')
    text2 = request.form.get('numero')
    side = site_name.split("/")[1]

    # Check if both text1 and text2 are provided
    if not text1 or not text2 or not site_name:
        return abort(404, description="nom numero et site requis")
    
    # Construct the path for the selected site folder
    site_folder = os.path.join(base_image_dir, site_name)
    # Check if the folder exists
    if os.path.exists(site_folder):
        # Look for files starting with 'back'
        twice_files = [f for f in os.listdir(site_folder) if f.startswith('twice')]
        if not twice_files:
            return "No file starting with 'back' found in the specified folder", 404

        # Use the first file found starting with 'back'
        twice_image_path = os.path.join(site_folder, twice_files[0])
        image = cv2.imread(twice_image_path)
    else:
        return "Site folder not found", 404

    # if 'file' not in request.files:
    #     # Load and process the default image if no file is provided
    #     if os.path.exists(default_image_path_twice):
    #         image = cv2.imread(default_image_path_twice)
    #     else:
    #         return "Default image not found", 404
    # else:
    #     file = request.files['file']
    #     if file.filename == '':
    #         return "No selected file", 400

    #     # Read the image file
    #     in_memory_file = BytesIO()
    #     file.save(in_memory_file)
    #     in_memory_file.seek(0)
    #     image = Image.open(in_memory_file)
    #     image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Process the image
    final_image = process_image_twice(image, text1, text2, side)

    # Convert the final image to bytes
    _, buffer = cv2.imencode('.png', final_image)
    final_image_bytes = BytesIO(buffer)

    return send_file(final_image_bytes, mimetype='image/png', as_attachment=True, download_name='processed_image.png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
