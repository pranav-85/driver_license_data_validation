from roboflow import Roboflow
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import shutil

rf = Roboflow(api_key="jJyjVbMQRppa3lA1JR0i")
project = rf.workspace().project("driver-s-license-text-extraction-6zjsk")
model = project.version(1).model

# infer on a local image
pred_data = model.predict("sample.jpg", confidence=40, overlap=30).json()

# visualize your prediction
model.predict("sample.jpg", confidence=40, overlap=30).save("prediction.jpg")

image = cv2.imread('sample.jpg')

# Convert the image to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Save the grayscale image
cv2.imwrite('sample_grayscale.jpg', grayscale_image)

def adjust_black_and_white(image_path, threshold=50):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Check if the image was loaded correctly
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None  # Exit the function if image loading fails

    # Create a mask for pixels close to black (i.e., pixels where all channels are close to 0)
    black_mask = np.all(image <= threshold, axis=-1)

    # Create a mask for pixels close to white (i.e., pixels where all channels are close to 255)
    white_mask = np.all(image >= 255 - threshold, axis=-1)

    # Set pixels close to black to pure black
    image[black_mask] = [0, 0, 0]

    # Set pixels close to white to pure white
    image[white_mask] = [255, 255, 255]

    return image


# Function to crop the image based on midpoint coordinates
def crop_image_by_midpoints(image_path, coordinates, output_dir):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over the coordinates and crop the image
    for i, coord in enumerate(coordinates):
        # Extract the midpoint coordinates (x, y), and width, height
        x = coord['x']
        y = coord['y']
        width = coord['width']
        height = coord['height']

        extra = 0
        # Calculate the top-left and bottom-right corners based on the midpoint and size
        x1 = int(x - width / 2) -extra
        y1 = int(y - height / 2)  - extra
        x2 = int(x + width / 2) + extra
        y2 = int(y + height / 2) + extra

        # Ensure that the coordinates are within the image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        # Crop the image using NumPy array slicing
        cropped_image = image[y1:y2, x1:x2]
        # Save the cropped image
        output_image_path = os.path.join(output_dir, f"{coord['class']}_{i+1}.jpg")
        cv2.imwrite(output_image_path, cropped_image)
        print(f"Saved cropped image: {output_image_path}")

        image_path = output_image_path  # Make sure this path is correct!

        # Adjust pixels close to black and white
        adjusted_image = adjust_black_and_white(image_path)

        if adjusted_image is not None:
        # Save the resulting image
            output_path = output_image_path
            cv2.imwrite(output_path, adjusted_image)

# Example coordinates (replace this with your actual annotation data)

# Path to the image file
image_path = 'sample_grayscale.jpg'

# Directory to save cropped images
output_dir = 'cropped_images'

# Call the function to crop the image based on midpoint coordinates
crop_image_by_midpoints(image_path, pred_data['predictions'], output_dir)


extract_classes = ['First name', 'Last name', 'License number', 'Exp date']

data = {}
# Check if the directory exists
if not os.path.isdir(cropped_images_dir):
    print(f"Error: The directory {cropped_images_dir} does not exist.")
else:
    # Load the TrOCR model and processor
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

    # Initialize a dictionary to store the extracted text class-wise
    extracted_text_dict = {}

    # Loop over each image file in the directory
    for filename in os.listdir(cropped_images_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(cropped_images_dir, filename)
            
            try:
                # Open the image
                image = Image.open(image_path).convert("RGB")
                
                # Process the image
                pixel_values = processor(images=image, return_tensors="pt").pixel_values

                # Generate text prediction
                generated_ids = model.generate(pixel_values, max_new_tokens=100)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # Extract class name from filename (modify as needed)
                class_name = filename.split('_')[0]  # Adjust if needed for different file naming

                # Store the extracted text in the dictionary
                if class_name not in extracted_text_dict:
                    extracted_text_dict[class_name] = []
                extracted_text_dict[class_name].append(generated_text.strip())

                # Optionally, print the result to the console
                print(f"Class: {class_name}")
                print(f"Extracted Text: {generated_text.strip()}")
                print("-" * 50)

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    # After all images are processed, print the dictionary
    print("\nExtracted Text Dictionary Class-Wise:")
    for class_name, texts in extracted_text_dict.items():
        print(f"Class: {class_name}")
        if class_name in extract_classes:
            data[class_name] = ""
        for text in texts:
            print(f"  Extracted Text: {text}")
            if class_name in extract_classes:
                data[class_name] += text
        print("-" * 50)

data['Name'] = data['First name'] +' ' + data['Last name']
del data['First name']
del data['Last name']


# Check if the directory exists
if os.path.exists(cropped_images_dir) and os.path.isdir(cropped_images_dir):
    try:
        # Remove the directory and all of its contents
        shutil.rmtree(cropped_images_dir)
        print(f"'{cropped_images_dir}' has been removed successfully.")
    except Exception as e:
        print(f"Error occurred while removing the directory: {e}")
else:
    print(f"The directory '{cropped_images_dir}' does not exist.")