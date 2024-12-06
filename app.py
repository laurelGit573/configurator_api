from flask import Flask, request, send_file, abort
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from io import BytesIO

app = Flask(__name__)

# Define your font path and default image path
font_path = "RunningYolo/fonts/AbingdonBold.otf"
default_image_path_back = "RunningYolo/Images/back.jpg"
default_image_path_front = "RunningYolo/Images/front.jpg"  # Specify the default image path

def process_image_front(image, text1):

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
                draw.text(centered_position, text, font=font, fill=(255, 255, 255))
                return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            finalImage = add_text_with_pillow(finalImage, text, (cX, cY - 95), font_path, 45)

    return finalImage

def process_image(image, text1, text2):
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
                draw.text(centered_position, text, font=font, fill=(255, 255, 255))
                return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            finalImage = add_text_with_pillow(finalImage, text1, (cX, cY - 95), font_path, 25)
            finalImage = add_text_with_pillow(finalImage, text2, (cX, cY - 20), font_path, 100)

    return finalImage

@app.route('/process/back', methods=['GET'])
def process():
    # Get text1 and text2 from the request
    text1 = request.form.get('nom')
    text2 = request.form.get('numero')

    # Check if both text1 and text2 are provided
    if not text1 or not text2:
        return abort(404, description="Text1 and Text2 must be provided")
    if 'file' not in request.files:
        # Load and process the default image if no file is provided
        if os.path.exists(default_image_path_back):
            image = cv2.imread(default_image_path_back)
        else:
            return "Default image not found", 404
    else:
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        # Read the image file
        in_memory_file = BytesIO()
        file.save(in_memory_file)
        in_memory_file.seek(0)
        image = Image.open(in_memory_file)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Process the image
    final_image = process_image(image, text1, text2)

    # Convert the final image to bytes
    _, buffer = cv2.imencode('.png', final_image)
    final_image_bytes = BytesIO(buffer)

    return send_file(final_image_bytes, mimetype='image/png', as_attachment=True, download_name='processed_image.png')


@app.route('/process/front', methods=['GET'])
def process_and_send_image_front():
    text1 = request.form.get('numero')
    if not text1:
        return abort(404, description="Text1 must be provided")
    if 'file' not in request.files:
        # Load and process the default image if no file is provided
        if os.path.exists(default_image_path_front):
            image = cv2.imread(default_image_path_front)
        else:
            return "Default image not found", 404
    else:
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        # Read the image file
        in_memory_file = BytesIO()
        file.save(in_memory_file)
        in_memory_file.seek(0)
        image = Image.open(in_memory_file)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
    processed_image_path = process_image_front(image, text1)
    # Convert the final image to bytes
    _, buffer = cv2.imencode('.png', processed_image_path)
    final_image_bytes = BytesIO(buffer)
    return send_file(final_image_bytes, mimetype='image/png',  as_attachment=True, download_name='processed_image_front.png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
