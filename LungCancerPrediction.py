# Import necessary libraries for Flask application
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from your React frontend

# Define constants
IMAGE_SIZE = (350, 350)
MODEL_PATH = 'C:/Users/Atman Mehta/HealthMate/HealthMate/models/Lung-cancer-prediction/models/trained_lung_cancer_model.h5'

# Class labels (make sure these match your trained model's classes)
class_labels = [
    'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib',
    'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa',
    'normal',
    'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'
]

# Load the pre-trained model
print("Loading model from:", MODEL_PATH)
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Function to load and preprocess an image for prediction
def preprocess_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image like the training images
    return img_array

@app.route('/predict_lung_cancer', methods=['POST'])
def predict_lung_cancer():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
            
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        # Read and process the image
        img = Image.open(file.stream)
        processed_img = preprocess_image(img, IMAGE_SIZE)
        
        # Make prediction
        predictions = model.predict(processed_img)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        predicted_label = class_labels[predicted_class]
        
        # Format response to match frontend expectations
        response = {
            'prediction': predicted_label,
            'confidence': confidence
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

# Route for testing if the server is up
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'Lung cancer prediction service is running'})

# For training the model (if needed)
def train_model():
    # Import statements for training
    import warnings
    warnings.filterwarnings('ignore')
    
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import LabelEncoder
    
    import tensorflow.keras
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras import utils
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
    
    # Define paths
    base_dir = r'C:\Users\Atman Mehta\HEALTHMATE\HealthMate\static\lung_data'
    train_folder = os.path.join(base_dir, 'train')
    test_folder = os.path.join(base_dir, 'test')
    validate_folder = os.path.join(base_dir, 'valid')
    
    # Initialize data generators
    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    batch_size = 8
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_folder,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        color_mode="rgb",
        class_mode='categorical'
    )
    
    validation_generator = test_datagen.flow_from_directory(
        test_folder,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        color_mode="rgb",
        class_mode='categorical'
    )
    
    # Set up callbacks
    learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=5, verbose=2, factor=0.5, min_lr=0.000001)
    early_stops = EarlyStopping(monitor='loss', min_delta=0, patience=6, verbose=2, mode='auto')
    checkpointer = ModelCheckpoint(filepath='best_model.hdf5', verbose=2, save_best_only=True, save_weights_only=True)
    
    # Define output size
    OUTPUT_SIZE = 4
    
    # Load pre-trained model
    pretrained_model = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
    pretrained_model.trainable = False
    
    # Create model
    model = Sequential()
    model.add(pretrained_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(OUTPUT_SIZE, activation='softmax'))
    
    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=25,
        epochs=50,
        callbacks=[learning_rate_reduction, early_stops, checkpointer],
        validation_data=validation_generator,
        validation_steps=20
    )
    
    # Save model
    model.save(MODEL_PATH)
    
    print("Model training completed and saved to", MODEL_PATH)
    return model

if __name__ == '__main__':
    # Check if model exists, if not train it
    if not os.path.exists(MODEL_PATH):
        print("Pre-trained model not found. Training new model...")
        model = train_model()
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5001, debug=True)