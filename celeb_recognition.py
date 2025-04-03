import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import requests
from io import BytesIO

# Configuration
IMG_SIZE = 224  # Increased from 160 to 224 for better feature extraction
BATCH_SIZE = 32
EPOCHS = 30  # Increased from 10 to 30
NUM_CLASSES = 5000  # Reduced from 10k to 5k to focus on more frequent identities
LEARNING_RATE = 1e-4  # Added explicit learning rate

# Paths
IMG_DIR = 'img_align_celeba'
IDENTITY_FILE = 'identity_CelebA.txt'
PARTITION_FILE = 'list_eval_partition.txt'
IDENTITY_NAME_FILE = 'list_identity_celeba.txt'

def load_identity_data():
    """Load celebrity identity information"""
    identities = {}
    with open(IDENTITY_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                img_name, identity = parts
                identities[img_name] = int(identity)
    return identities

def load_partition_data():
    """Load train/val/test split information"""
    partitions = {}
    with open(PARTITION_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                img_name, partition = parts
                partitions[img_name] = int(partition)
    return partitions

def load_celebrity_names():
    """Load celebrity names from list_identity_celeba.txt file - Optimized version"""
    print("Loading celebrity names...")
    
    # Dictionary to map identity IDs to names
    identity_to_name = {}
    
    # First, create an efficient lookup for image names to identity IDs
    image_to_identity = {}
    try:
        print("Creating image to identity mapping...")
        with open(IDENTITY_FILE, 'r') as id_file:
            for line in id_file:
                parts = line.strip().split()
                if len(parts) == 2:
                    img_name, identity_id = parts
                    image_to_identity[img_name] = int(identity_id)
        print(f"Created mapping for {len(image_to_identity)} images")
    except Exception as e:
        print(f"Error creating image to identity mapping: {e}")
        return {}
    
    # Now process the identity names file
    try:
        print("Processing identity names file...")
        with open(IDENTITY_NAME_FILE, 'r') as f:
            lines = f.readlines()
            
            # Skip the first line (count) and check format
            if len(lines) < 3:
                print("Identity file format is not as expected")
                return {}
            
            # Skip header lines
            for i, line in enumerate(lines[2:], 2):
                if i % 10000 == 0:
                    print(f"Processed {i} names...")
                    
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    # Name may contain underscores, join everything after the image name
                    name = ' '.join(parts[1:])
                    name = name.replace('_', ' ')  # Replace underscores with spaces
                    
                    # Get the identity ID from our pre-built mapping
                    identity_id = image_to_identity.get(img_name)
                    if identity_id is not None:
                        identity_to_name[identity_id] = name
    except Exception as e:
        print(f"Error processing identity names: {e}")
        return {}
    
    print(f"Loaded {len(identity_to_name)} celebrity names")
    return identity_to_name

def prepare_dataset():
    """Prepare dataset for training"""
    print("Loading identity and partition data...")
    identities = load_identity_data()
    partitions = load_partition_data()
    
    # Create DataFrame for easier manipulation
    data = []
    for img_name, identity in identities.items():
        if img_name in partitions:
            data.append({
                'image': img_name,
                'identity': identity,
                'partition': partitions[img_name]
            })
    
    df = pd.DataFrame(data)
    
    # Get top N most frequent identities
    identity_counts = df['identity'].value_counts()
    top_identities = identity_counts.head(NUM_CLASSES).index.tolist()
    
    # Filter dataset to include only these identities
    df_filtered = df[df['identity'].isin(top_identities)]
    
    # Create a mapping from identity ID to a sequential class index
    identity_to_label = {identity: i for i, identity in enumerate(top_identities)}
    df_filtered['label'] = df_filtered['identity'].map(identity_to_label)
    
    # Save the mapping for later use when predicting
    np.save('identity_mapping.npy', identity_to_label)
    
    # Try to load celebrity names
    try:
        celebrity_names = load_celebrity_names()
        # Create a mapping from label to name
        label_to_name = {identity_to_label[identity]: celebrity_names.get(identity, f"Unknown-{identity}") 
                        for identity in top_identities if identity in identity_to_label}
        np.save('name_mapping.npy', label_to_name)
        print("Celebrity name mapping saved successfully")
    except Exception as e:
        print(f"Error loading celebrity names: {e}. Continuing without names.")
    
    print(f"Dataset prepared with {len(df_filtered)} images and {len(top_identities)} unique identities")
    return df_filtered

def create_model(num_classes):
    """Create a fine-tuned model for celebrity recognition with improved architecture"""
    # Use ResNet50V2 as the base model for better feature extraction
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Fine-tune the top layers of the base model
    for layer in base_model.layers[:-30]:  # Freeze earlier layers, train later layers
        layer.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Use a better optimizer with explicit learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

class CelebDataGenerator(tf.keras.utils.Sequence):
    """Custom data generator for loading and preprocessing celebrity images with augmentation"""
    def __init__(self, dataframe, img_dir, batch_size=32, img_size=(224, 224), shuffle=True, augment=False):
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(dataframe))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
        # Create augmentation generator if needed
        if self.augment:
            self.augmenter = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode='nearest'
            )
    
    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.dataframe.iloc[batch_indexes]
        
        batch_x = []
        batch_y = []
        
        for _, row in batch_df.iterrows():
            img_path = os.path.join(self.img_dir, row['image'])
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
                
                # Apply face detection and alignment for better results
                # (simplified version - in production would use more advanced face alignment)
                try:
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    if len(faces) > 0:
                        x, y, w, h = faces[0]  # Take the first face
                        # Add some margin
                        margin = int(0.1 * max(w, h))
                        x = max(0, x - margin)
                        y = max(0, y - margin)
                        w = min(img.shape[1] - x, w + 2 * margin)
                        h = min(img.shape[0] - y, h + 2 * margin)
                        
                        # Crop to face area
                        img = img[y:y+h, x:x+w]
                        img = cv2.resize(img, self.img_size)
                except Exception as e:
                    # If face detection fails, use the whole image
                    pass
                
                # Normalize to [0,1]
                img = img / 255.0
                
                # Apply augmentation if needed
                if self.augment:
                    img = self.augmenter.random_transform(img)
                
                batch_x.append(img)
                batch_y.append(row['label'])
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        
        if not batch_x:  # If batch is empty
            # Provide a default empty batch to avoid errors
            return np.empty((0, *self.img_size, 3)), np.empty(0)
        
        return np.array(batch_x), np.array(batch_y)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def train_model():
    """Train the celebrity recognition model with improved training strategy"""
    df = prepare_dataset()
    
    # Split into train, validation, and test sets based on the predefined partitions
    train_df = df[df['partition'] == 0]
    val_df = df[df['partition'] == 1]
    test_df = df[df['partition'] == 2]
    
    print(f"Train set: {len(train_df)} images")
    print(f"Validation set: {len(val_df)} images")
    print(f"Test set: {len(test_df)} images")
    
    # Create data generators with augmentation for training
    train_generator = CelebDataGenerator(train_df, IMG_DIR, batch_size=BATCH_SIZE, img_size=(IMG_SIZE, IMG_SIZE), augment=True)
    val_generator = CelebDataGenerator(val_df, IMG_DIR, batch_size=BATCH_SIZE, img_size=(IMG_SIZE, IMG_SIZE), shuffle=False)
    
    # Create and train the model
    model = create_model(NUM_CLASSES)
    
    # Print model summary
    model.summary()
    
    # Add callbacks for better training
    checkpoint = ModelCheckpoint(
        'best_model.h5', 
        monitor='val_accuracy', 
        verbose=1, 
        save_best_only=True,
        mode='max'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1
    )
    
    # Load the best model
    model = tf.keras.models.load_model('best_model.h5')
    
    # Save the final model
    model.save('celebrity_recognition_model.h5')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    return model, history

def evaluate_model(model, test_df):
    """Evaluate the model on the test set"""
    test_generator = CelebDataGenerator(test_df, IMG_DIR, batch_size=BATCH_SIZE, img_size=(IMG_SIZE, IMG_SIZE), shuffle=False)
    
    # Evaluate the model
    results = model.evaluate(test_generator, verbose=1)
    print(f"Test Loss: {results[0]}")
    print(f"Test Accuracy: {results[1]}")

def preprocess_image(image_path_or_url):
    """Preprocess an image for prediction with improved face detection"""
    try:
        if image_path_or_url.startswith(('http://', 'https://')):
            response = requests.get(image_path_or_url)
            img = Image.open(BytesIO(response.content))
            img = np.array(img)
        else:
            img = cv2.imread(image_path_or_url)
            if img is None:
                print(f"Error: Could not read image {image_path_or_url}")
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Try face detection for better results
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]  # Take the first face
                # Add margin
                margin = int(0.1 * max(w, h))
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(img.shape[1] - x, w + 2 * margin)
                h = min(img.shape[0] - y, h + 2 * margin)
                
                # Crop to face area
                img = img[y:y+h, x:x+w]
        except Exception as e:
            print(f"Face detection error (using full image): {e}")
        
        # Resize and preprocess
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0  # Normalize
        
        return np.expand_dims(img, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_celebrity(model, image_path_or_url):
    """Predict the identity of a celebrity in a given image with name information"""
    # Load identity mapping
    identity_mapping = np.load('identity_mapping.npy', allow_pickle=True).item()
    reverse_mapping = {v: k for k, v in identity_mapping.items()}
    
    # Try to load name mapping
    try:
        name_mapping = np.load('name_mapping.npy', allow_pickle=True).item()
    except Exception as e:
        print(f"Name mapping not found: {e}. Using numeric IDs.")
        name_mapping = {}
    
    # Preprocess the image
    img = preprocess_image(image_path_or_url)
    if img is None:
        return None
    
    # Make prediction
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    # Get the celebrity ID and name if available
    celebrity_id = reverse_mapping[predicted_class]
    celebrity_name = name_mapping.get(predicted_class, f"Unknown-{celebrity_id}")
    
    # Create top predictions with names if available
    top_predictions = []
    for i in np.argsort(-predictions[0])[:5]:  # Top 5 predictions
        pred_id = reverse_mapping[i]
        pred_name = name_mapping.get(i, f"Unknown-{pred_id}")
        
        top_predictions.append({
            'celebrity_id': pred_id,
            'celebrity_name': pred_name,
            'confidence': float(predictions[0][i])
        })
    
    return {
        'celebrity_id': celebrity_id,
        'celebrity_name': celebrity_name,
        'confidence': float(confidence),
        'top_predictions': top_predictions
    }

def main():
    """Main function to train the model and make predictions"""
    # Check if the model already exists
    if os.path.exists('celebrity_recognition_model.h5'):
        print("Loading existing model...")
        model = tf.keras.models.load_model('celebrity_recognition_model.h5')
        df = prepare_dataset()
        test_df = df[df['partition'] == 2]
    else:
        print("Training new model...")
        model, _ = train_model()
        df = prepare_dataset()
        test_df = df[df['partition'] == 2]
        evaluate_model(model, test_df)
    
    # Example of predicting a celebrity from a URL
    image_url = input("Enter a URL to an image of a celebrity: ")
    result = predict_celebrity(model, image_url)
    
    if result:
        print(f"Predicted Celebrity: {result['celebrity_name']} (ID: {result['celebrity_id']})")
        print(f"Confidence: {result['confidence']:.2f}")
        print("\nTop 5 Predictions:")
        for i, pred in enumerate(result['top_predictions']):
            print(f"{i+1}. {pred['celebrity_name']} (ID: {pred['celebrity_id']}), Confidence: {pred['confidence']:.2f}")

if __name__ == "__main__":
    main() 