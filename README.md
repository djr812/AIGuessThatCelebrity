# Celebrity Recognition App

This application uses TensorFlow to learn facial characteristics from the CelebA dataset and identify celebrities from images.

## Dataset

The application uses the CelebA dataset, which includes:
- 202,599 face images of celebrities
- Identity information for each image
- Train/validation/test partition information

## Features

- **Deep Learning Model**: Uses a fine-tuned MobileNetV2 architecture to identify celebrities.
- **Web Interface**: User-friendly web interface for uploading images or providing URLs.
- **Real-time Predictions**: Get instant predictions when you submit an image.
- **Top-K Predictions**: View the most likely identities with confidence scores.

## Requirements

The project requires the following dependencies:
- Python 3.6+
- TensorFlow 2.x
- Flask
- OpenCV
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd celebrity-recognition
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Make sure you have the CelebA dataset in the project directory:
   - `img_align_celeba/`: Directory containing celebrity face images
   - `identity_CelebA.txt`: File mapping images to identity IDs
   - `list_eval_partition.txt`: File specifying train/val/test splits

## Usage

### Training the Model

To train the celebrity recognition model, run:
```
python celeb_recognition.py
```

This will:
1. Load and process the identity and partition data
2. Filter to the top 10,000 most frequent identities
3. Create and train a deep learning model
4. Save the trained model to `celebrity_recognition_model.h5`

### Running the Web App

To start the web application, run:
```
python app.py
```

Then open your browser and navigate to `http://localhost:5000`.

The web interface allows you to:
- Upload an image file from your computer
- Provide a URL to an online image
- View the predicted celebrity ID and confidence score
- See alternative predictions with their respective confidence scores

## Model Architecture

The model uses transfer learning with MobileNetV2 as the base model:
- Pre-trained on ImageNet for feature extraction
- Global average pooling to reduce spatial dimensions
- Dense layers for classification
- Training with data generators for efficient memory usage

## Limitations

- The model identifies celebrities by their numeric ID rather than names, as the dataset doesn't provide name information.
- Performance may vary based on image quality, lighting, angles, and other factors.
- The model is limited to identifying celebrities included in the CelebA dataset.

## Future Improvements

- Add a mapping from ID to celebrity names if such data becomes available
- Implement face detection to handle images with multiple people
- Add face alignment preprocessing to improve accuracy
- Create a more sophisticated feature extraction pipeline
- Support for model retraining with new data

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The CelebA dataset is provided by the Multimedia Laboratory, The Chinese University of Hong Kong. 