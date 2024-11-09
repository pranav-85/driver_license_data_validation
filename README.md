# Driver's License OCR Text Extraction Web Application

This project is an OCR-driven application for extracting key information from driver's licenses, including fields like first name, last name, license number, and expiration date. The web app is built using **React** for the frontend and **Flask** for the backend, leveraging **Roboflow** for object detection and **Hugging Face's TrOCR** for OCR processing.

## Features

- **Automated OCR extraction** for text fields on driver's licenses.
- **Image preprocessing** with brightness adjustments, grayscale conversion, and black/white contrast enhancement to improve OCR accuracy.
- **React and Flask-based Web App** that provides an intuitive UI for uploading images and displaying extracted text.

---

## Dataset

The **Driver's License Text Extraction Dataset** used to train the object detection model is sourced from [Roboflow]([https://roboflow.com/](https://universe.roboflow.com/amyf467-gmail-com/driving-license-dnmj8/dataset/3)). This dataset was used to identify and locate text fields within license images.

## Models and Libraries Used

### 1. **Roboflow**
   - **Roboflow API**: Used for object detection on driver's license images to segment specific text fields.

### 2. **Hugging Face TrOCR**
   - **TrOCRProcessor** and **VisionEncoderDecoderModel**: Transformer-based OCR model for accurate text extraction from segmented fields.

### 3. **OpenCV**
   - Image preprocessing tasks, such as grayscale conversion, brightness adjustments, and bounding box cropping, are handled with OpenCV.

---

## Installation

1. **Clone the repository** and navigate to the project directory:
   ```bash
   git clone https://github.com/yourusername/license-text-extraction.git
   cd license-text-extraction
  
2. **Install Python Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up Roboflow API Key**
    ```python
    rf = Roboflow(api_key="YOUR_API_KEY")
    ```
4. **Start the Flask Backend**
   ```bash
   python app.py
   ```
5. **Set up React Frontend**
   ```bash
   cd app
   npm install
   npm start
   ```
6. **Access the web application**
   Open your browser and go to `http://localhost:3000` to access the application.
