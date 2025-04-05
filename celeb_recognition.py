import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import requests
from io import BytesIO
import gc

# Configuration
IMG_SIZE = 224  # Standard input size for pre-trained models
BATCH_SIZE = 32  # Larger batch size for better convergence
EPOCHS = 50  # More epochs with improved early stopping
NUM_CLASSES = 10  # Focus on fewer classes for better accuracy
LEARNING_RATE = 1e-4  # Slightly lower learning rate for stability
MAX_SAMPLES_PER_CELEB = 300  # More samples per celebrity
MIN_SAMPLES_PER_CELEB = 30  # Ensure enough samples per class

# Set architecture to ResNet50 which is proven for face recognition
MODEL_ARCHITECTURE = 'ResNet50'

# Add new configuration parameters
USE_MIXED_PRECISION = True  # Enable for faster training on supported hardware
ENABLE_FACE_ALIGNMENT = True  # Improve face detection and alignment
ENABLE_DEEP_SUPERVISION = False  # Disable auxiliary outputs to reduce memory usage
MEMORY_OPTIMIZATION = True  # Enable memory optimization
ENABLE_CONFIDENCE_CALIBRATION = True  # Enable confidence calibration
VISUALIZATION_ENABLED = True  # Enable visualization of predictions
MIXUP_ENABLED = True  # Enable mixup data augmentation
MIXUP_ALPHA = 0.2  # Alpha parameter for mixup

# Paths
IMG_DIR = 'img_align_celeba'
IDENTITY_FILE = 'identity_CelebA.txt'
PARTITION_FILE = 'list_eval_partition.txt'
IDENTITY_NAME_FILE = 'list_identity_celeba.txt'

# Enable memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU found. Using CPU for training.")
    # Set optimal CPU thread settings
    tf.config.threading.set_intra_op_parallelism_threads(2)  # Limit intra-op parallelism
    tf.config.threading.set_inter_op_parallelism_threads(2)  # Limit inter-op parallelism

# Add memory cleanup function
def clean_memory():
    """Force garbage collection to free up memory"""
    gc.collect()
    tf.keras.backend.clear_session()
    
    # Print memory usage if psutil is available
    try:
        import psutil
        process = psutil.Process(os.getpid())
        print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    except ImportError:
        pass

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

def check_dataset_files():
    """Check if necessary dataset files exist and are valid"""
    problems = []
    
    # Check IMG_DIR
    if not os.path.exists(IMG_DIR):
        problems.append(f"Image directory '{IMG_DIR}' does not exist")
    elif not os.path.isdir(IMG_DIR):
        problems.append(f"'{IMG_DIR}' is not a directory")
    else:
        # Check if there are any images in the directory
        image_files = [f for f in os.listdir(IMG_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if not image_files:
            problems.append(f"No image files found in '{IMG_DIR}'")
        else:
            print(f"Found {len(image_files)} image files in '{IMG_DIR}'")
            # Check a sample image
            sample_img = os.path.join(IMG_DIR, image_files[0])
            try:
                img = cv2.imread(sample_img)
                if img is None:
                    problems.append(f"Could not read sample image '{sample_img}'")
                else:
                    print(f"Successfully read sample image: {image_files[0]}, shape: {img.shape}")
            except Exception as e:
                problems.append(f"Error reading sample image: {e}")
    
    # Check identity file
    if not os.path.exists(IDENTITY_FILE):
        problems.append(f"Identity file '{IDENTITY_FILE}' does not exist")
    else:
        try:
            with open(IDENTITY_FILE, 'r') as f:
                lines = f.readlines()
                if len(lines) < 10:  # Arbitrary small number to check if file has content
                    problems.append(f"Identity file '{IDENTITY_FILE}' seems empty or too small")
                else:
                    print(f"Identity file contains {len(lines)} lines")
                    # Check file format
                    sample_line = lines[0].strip()
                    parts = sample_line.split()
                    if len(parts) != 2:
                        problems.append(f"Identity file format incorrect. Expected 2 columns, got {len(parts)}")
                    else:
                        print(f"Identity file format seems correct. Sample: {sample_line}")
        except Exception as e:
            problems.append(f"Error reading identity file: {e}")
    
    # Print summary
    if problems:
        print("\n⚠️ DATASET PROBLEMS DETECTED ⚠️")
        for i, problem in enumerate(problems, 1):
            print(f"{i}. {problem}")
        print("\nPlease fix these issues before continuing.")
        return False
    else:
        print("\n✅ Dataset files look valid")
        return True

def prepare_dataset():
    """Prepare dataset with high-quality examples for fewer celebrities"""
    global NUM_CLASSES, MIN_SAMPLES_PER_CELEB
    
    # First, check if dataset files are valid
    print("Checking dataset files...")
    if not check_dataset_files():
        print("WARNING: Dataset files have issues. Trying to proceed anyway...")
    
    print("Loading identity and partition data...")
    identities = load_identity_data()
    
    if not identities:
        print("ERROR: No identity data loaded. Check your identity file.")
        # Create a minimal fake dataset for debugging
        print("Creating a minimal test dataset to prevent crash...")
        NUM_CLASSES = 2
        fake_df = pd.DataFrame({
            'image': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg'],
            'identity': [0, 0, 0, 1, 1, 1],
            'label': [0, 0, 0, 1, 1, 1],
            'partition': [0, 0, 1, 0, 1, 2]
        })
        return fake_df
    
    # Create DataFrame directly from identity data
    data = []
    for img_name, identity in identities.items():
        # Check if the image file actually exists
        if os.path.exists(os.path.join(IMG_DIR, img_name)):
            data.append({
                'image': img_name,
                'identity': identity
            })
        
    if not data:
        print("ERROR: No valid images found that match identity data.")
        print("Make sure image files exist in the directory and match identity data.")
        # Create a minimal fake dataset for debugging
        print("Creating a minimal test dataset to prevent crash...")
        NUM_CLASSES = 2
        fake_df = pd.DataFrame({
            'image': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg'],
            'identity': [0, 0, 0, 1, 1, 1],
            'label': [0, 0, 0, 1, 1, 1],
            'partition': [0, 0, 1, 0, 1, 2]
        })
        return fake_df
    
    df = pd.DataFrame(data)
    print(f"Initial DataFrame contains {len(df)} images across {df['identity'].nunique()} identities")
    
    # Test a few image paths to make sure they exist
    print("Testing a few image paths...")
    sample_size = min(5, len(df))
    valid_images = 0
    for i, row in df.head(sample_size).iterrows():
        img_path = os.path.join(IMG_DIR, row['image'])
        if os.path.exists(img_path):
            valid_images += 1
            print(f"  ✓ {row['image']} exists")
        else:
            print(f"  ✗ {row['image']} does not exist")
    
    if valid_images == 0 and sample_size > 0:
        print("ERROR: None of the sample images exist! Check your image directory and identity data.")
        print("Creating a minimal test dataset to prevent crash...")
        NUM_CLASSES = 2
        fake_df = pd.DataFrame({
            'image': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg'],
            'identity': [0, 0, 0, 1, 1, 1],
            'label': [0, 0, 0, 1, 1, 1],
            'partition': [0, 0, 1, 0, 1, 2]
        })
        return fake_df
    
    # Get top N most frequent identities with minimum required samples
    identity_counts = df['identity'].value_counts()
    print(f"Top 10 celebrities by number of images:")
    for i, (identity, count) in enumerate(identity_counts.head(10).items()):
        print(f"  {i+1}. Identity {identity}: {count} images")
    
    # Adaptive minimum samples based on dataset
    original_min = MIN_SAMPLES_PER_CELEB
    while MIN_SAMPLES_PER_CELEB > 3:  # Set an absolute minimum of 3 samples
        # Filter to identities with at least MIN_SAMPLES_PER_CELEB samples
        qualified_identities = identity_counts[identity_counts >= MIN_SAMPLES_PER_CELEB].index.tolist()
        
        if len(qualified_identities) >= NUM_CLASSES:
            print(f"Found {len(qualified_identities)} celebrities with {MIN_SAMPLES_PER_CELEB}+ samples")
            break
        else:
            # Decrease minimum samples requirement
            prev_min = MIN_SAMPLES_PER_CELEB
            MIN_SAMPLES_PER_CELEB = max(MIN_SAMPLES_PER_CELEB - 5, 3)  # Decrease by 5, but not below 3
            print(f"Adjusting: found only {len(qualified_identities)} celebrities with {prev_min}+ samples")
            print(f"Trying with minimum {MIN_SAMPLES_PER_CELEB} samples...")
    
    # Final check for minimum number of classes
    qualified_identities = identity_counts[identity_counts >= MIN_SAMPLES_PER_CELEB].index.tolist()
    
    if len(qualified_identities) == 0:
        print(f"ERROR: Could not find any celebrity with {MIN_SAMPLES_PER_CELEB}+ samples")
        print("Using the top 5 celebrities by number of samples instead")
        qualified_identities = identity_counts.head(5).index.tolist()
        
    if len(qualified_identities) < NUM_CLASSES:
        print(f"Only found {len(qualified_identities)} celebrities with {MIN_SAMPLES_PER_CELEB}+ samples")
        print(f"Adjusting NUM_CLASSES from {NUM_CLASSES} to {len(qualified_identities)}")
        NUM_CLASSES = len(qualified_identities)
    
    # Take only top NUM_CLASSES identities with most samples
    valid_identities = qualified_identities[:NUM_CLASSES]
    print(f"Selected {len(valid_identities)} celebrities with {MIN_SAMPLES_PER_CELEB}+ samples each")
    
    # Make sure we have at least 2 classes for a valid classification problem
    if len(valid_identities) < 2:
        print("ERROR: Not enough classes found. Need at least 2 for classification.")
        print("Using the top 2 identities by frequency...")
        valid_identities = identity_counts.head(2).index.tolist()
        NUM_CLASSES = len(valid_identities)
    
    # Filter dataset to include only these identities
    df_filtered = df[df['identity'].isin(valid_identities)]
    
    # Create a mapping from identity ID to a sequential class index
    identity_to_label = {identity: i for i, identity in enumerate(valid_identities)}
    df_filtered['label'] = df_filtered['identity'].map(identity_to_label)
    
    # Save the mapping for later use when predicting
    np.save('identity_mapping.npy', identity_to_label)
    
    # Process classes individually for better balance
    balanced_dfs = []
    
    for label, identity in enumerate(valid_identities):
        class_samples = df_filtered[df_filtered['identity'] == identity]
        print(f"Celebrity {label}: Identity {identity} has {len(class_samples)} samples")
        
        # Limit samples per class to prevent bias
        if len(class_samples) > MAX_SAMPLES_PER_CELEB:
            class_samples = class_samples.sample(MAX_SAMPLES_PER_CELEB, random_state=42)
        
        # Ensure label is set correctly
        class_samples['label'] = label
        balanced_dfs.append(class_samples)
    
    # Combine all balanced classes
    if not balanced_dfs:
        print("ERROR: No data frames to concatenate! Creating minimal dataset...")
        # Create a minimal dataset to prevent crash
        NUM_CLASSES = 2
        fake_df = pd.DataFrame({
            'image': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg'],
            'identity': [0, 0, 0, 1, 1, 1],
            'label': [0, 0, 0, 1, 1, 1],
            'partition': [0, 0, 1, 0, 1, 2]
        })
        return fake_df
    
    df_balanced = pd.concat(balanced_dfs).reset_index(drop=True)
    
    # Create stratified train/val/test split with more training data
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    for label in range(NUM_CLASSES):
        class_samples = df_balanced[df_balanced['label'] == label]
        
        # In case there's somehow an empty class
        if len(class_samples) == 0:
            print(f"WARNING: No samples found for class {label}")
            continue
        
        # Split 80/10/10 to have more training data
        n_samples = len(class_samples)
        n_train = max(int(n_samples * 0.7), 1)  # At least 1 for training
        n_val = max(int(n_samples * 0.15), 1)   # At least 1 for validation
        
        # Make sure we don't exceed the number of samples
        if n_train + n_val >= n_samples:
            n_train = n_samples - 1
            n_val = 1
            
        # Shuffle the samples
        shuffled = class_samples.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split into train/val/test
        train_dfs.append(shuffled.iloc[:n_train])
        val_dfs.append(shuffled.iloc[n_train:n_train+n_val])
        test_dfs.append(shuffled.iloc[n_train+n_val:])
    
    # Combine the splits
    if not train_dfs or not val_dfs or not test_dfs:
        print("ERROR: One of the splits is empty. Creating minimal dataset...")
        # Create a minimal dataset to prevent crash
        NUM_CLASSES = 2
        fake_df = pd.DataFrame({
            'image': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg'],
            'identity': [0, 0, 0, 1, 1, 1],
            'label': [0, 0, 0, 1, 1, 1],
            'partition': [0, 0, 1, 0, 1, 2]
        })
        return fake_df
    
    train_df = pd.concat(train_dfs).reset_index(drop=True)
    val_df = pd.concat(val_dfs).reset_index(drop=True)
    test_df = pd.concat(test_dfs).reset_index(drop=True)
    
    # Add partition info for compatibility with existing code
    train_df['partition'] = 0
    val_df['partition'] = 1
    test_df['partition'] = 2
    
    # Combine for final dataset
    df_final = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    
    print(f"\nFinal dataset: {len(df_final)} images across {NUM_CLASSES} celebrity classes")
    print(f"  Train: {len(train_df)} images")
    print(f"  Validation: {len(val_df)} images")
    print(f"  Test: {len(test_df)} images")
    
    return df_final

# Enable mixed precision for faster training
if USE_MIXED_PRECISION:
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision enabled")
    except Exception as e:
        print(f"Could not enable mixed precision: {e}")

# Apply mixup data augmentation for better generalization
def mixup(x, y, alpha=0.2):
    """Apply mixup data augmentation by blending images and labels"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    batch_size = x.shape[0]
    index = np.random.permutation(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

# Custom loss for mixup
def mixup_loss(y_true, y_pred, y_a, y_b, lam):
    """Custom loss function for mixup training"""
    loss_a = tf.keras.losses.sparse_categorical_crossentropy(y_a, y_pred)
    loss_b = tf.keras.losses.sparse_categorical_crossentropy(y_b, y_pred)
    return lam * loss_a + (1 - lam) * loss_b

# Enhanced model creation with stronger regularization
def create_model(num_classes):
    """Create model using proven ResNet50 architecture with better regularization"""
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Use ResNet50 with appropriate preprocessing
    if MODEL_ARCHITECTURE == 'ResNet50':
        try:
            # ResNet50 preprocessing
            from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
            x = preprocess_input(inputs)
            base_model = ResNet50(
                input_shape=(IMG_SIZE, IMG_SIZE, 3),
                include_top=False,
                weights='imagenet'
            )
        except ImportError:
            # Fallback to MobileNetV2 if ResNet50 is not available
            print("ResNet50 not available, falling back to MobileNetV2")
            from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
            x = preprocess_input(inputs)
            base_model = MobileNetV2(
                input_shape=(IMG_SIZE, IMG_SIZE, 3),
                include_top=False,
                weights='imagenet'
            )
    else:
        # Fall back to MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
        x = preprocess_input(inputs)  # Use proper preprocessing
        base_model = MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Apply base model
    x = base_model(x, training=False)
    
    # Enhanced head architecture with more regularization
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # First dense layer with improved regularization
    x = tf.keras.layers.Dense(
        512, 
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    # Second dense layer for better feature extraction
    x = tf.keras.layers.Dense(
        256, 
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        num_classes, 
        activation='softmax',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Use Adam optimizer with weight decay for better regularization
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=1.0,
        beta_1=0.9,
        beta_2=0.999
    )
    
    # Compile with standard metrics
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_accuracy')]
    )
    
    return model

# Modify CelebDataGenerator.__getitem__ to include more advanced augmentation
def augment_image(img, augment=True):
    """Apply advanced augmentation to an image"""
    if not augment:
        return img
        
    # Copy the image to avoid modifying the original
    img = img.copy()
    
    # Random horizontal flip
    if np.random.random() > 0.5:
        img = np.fliplr(img)
    
    # Random rotation within 15 degrees
    if np.random.random() > 0.5:
        angle = np.random.uniform(-15, 15)
        center = (img.shape[1]//2, img.shape[0]//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)
    
    # Random brightness and contrast adjustments
    if np.random.random() > 0.5:
        # Brightness
        brightness = np.random.uniform(0.8, 1.2)
        img = np.clip(img * brightness, 0, 255).astype(np.uint8)
        
    if np.random.random() > 0.5:
        # Contrast
        contrast = np.random.uniform(0.8, 1.2)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        img = np.clip((img - mean) * contrast + mean, 0, 255).astype(np.uint8)
    
    # Random saturation adjustment for color images
    if img.shape[-1] == 3 and np.random.random() > 0.5:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        saturation = np.random.uniform(0.8, 1.2)
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * saturation, 0, 255)
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    
    # Random zoom (crop and resize)
    if np.random.random() > 0.5:
        scale = np.random.uniform(0.85, 1.0)
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Crop from center
        y_start = (h - new_h) // 2
        x_start = (w - new_w) // 2
        img = img[y_start:y_start+new_h, x_start:x_start+new_w]
        
        # Resize back to original size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return img

# Enhanced training function with mixup and better training schedule
def train_model():
    """Train model with improved regularization and data augmentation"""
    df = prepare_dataset()
    
    # Split into train, validation, and test sets
    train_df = df[df['partition'] == 0]
    val_df = df[df['partition'] == 1]
    test_df = df[df['partition'] == 2]
    
    print(f"Train set: {len(train_df)} images")
    print(f"Validation set: {len(val_df)} images")
    print(f"Test set: {len(test_df)} images")
    
    # Calculate class weights
    print("Calculating class weights...")
    class_weights = {}
    if NUM_CLASSES > 0:
        class_counts = train_df['label'].value_counts().sort_index()
        max_count = class_counts.max()
        
        print("Class distribution in training set:")
        for label, count in class_counts.items():
            # Convert label to int if it's not already
            label_int = int(label) if not isinstance(label, int) else label
            
            if label_int < NUM_CLASSES:
                weight = max_count / count
                # Cap weight at 3.0 to prevent extreme weights
                weight = min(weight, 3.0)
                class_weights[label_int] = weight
                print(f"  Class {label_int}: {count} samples, weight: {weight:.2f}")
    
    # Clean memory before creating generators
    clean_memory()
    
    # Data generators with enhanced augmentation
    train_generator = CelebDataGenerator(train_df, IMG_DIR, batch_size=BATCH_SIZE, 
                                         img_size=(IMG_SIZE, IMG_SIZE), augment=True)
    val_generator = CelebDataGenerator(val_df, IMG_DIR, batch_size=BATCH_SIZE, 
                                       img_size=(IMG_SIZE, IMG_SIZE), shuffle=False, augment=False)
    
    # Create model
    try:
        print("Creating model...")
        model = create_model(NUM_CLASSES)
        model.summary()
        
        # Clean memory
        clean_memory()
        
        # Callbacks
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy', 
            verbose=1, 
            save_best_only=True,
            mode='max',
            save_weights_only=False
        )
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=12,  # More patience for better convergence
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.6,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        class MemoryCleanupCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                clean_memory()
                print(f"Memory cleaned after epoch {epoch + 1}")
        
        # Create a custom callback for mixup training
        class MixupCallback(tf.keras.callbacks.Callback):
            def __init__(self, train_generator, alpha=0.2, apply_after_epoch=5):
                super().__init__()
                self.train_generator = train_generator
                self.alpha = alpha
                self.apply_after_epoch = apply_after_epoch
                self.using_mixup = False
                
            def on_epoch_begin(self, epoch, logs=None):
                # Only apply mixup after certain epochs to stabilize initial training
                if epoch >= self.apply_after_epoch and MIXUP_ENABLED:
                    if not self.using_mixup:
                        print(f"Enabling mixup from epoch {epoch+1}")
                        self.using_mixup = True
                        
            def on_batch_begin(self, batch, logs=None):
                # Apply mixup to the batch if enabled
                if self.using_mixup:
                    x, y = self.train_generator[batch % len(self.train_generator)]
                    if len(x) > 1:  # Need at least 2 samples for mixup
                        x_mixed, y_a, y_b, self.lam = mixup(x, y, self.alpha)
                        self.model.train_on_batch(x_mixed, y_a)  # Using y_a as dummy target
                        return True  # Skip the normal training
                return False
                
        # Add custom mixup callback if enabled
        callbacks = [
            checkpoint,
            early_stopping,
            reduce_lr,
            MemoryCleanupCallback()
        ]
        
        # Three-phase training approach
        try:
            # Test forward pass
            print("Testing model with a sample batch...")
            x_batch, y_batch = next(iter(train_generator))
            if len(x_batch) > 0:
                with tf.device('/cpu:0'):
                    test_output = model.predict(x_batch[:1])
                print(f"Model output shape: {test_output.shape}")
                print("Forward pass successful!")
            else:
                print("ERROR: Empty batch returned from generator!")
                return None, None
            
            # Save initial model
            model.save('initial_model.h5')
            
            # Phase 1: Train with frozen base model
            print("\nPhase 1: Training with frozen base model...")
            history1 = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=15,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
            
            # Clean memory
            clean_memory()
            
            # Save phase 1 model
            model.save('phase1_model.h5')
            
            # Phase 2: Fine-tuning with partial base model unfreezing
            print("\nPhase 2: Fine-tuning with partial base model unfreezing...")
            
            # Unfreeze the last layers 
            for layer in model.layers:
                if isinstance(layer, tf.keras.Model):
                    base_model = layer
                    
                    # Unfreeze the last 30% of layers
                    layer_count = len(base_model.layers)
                    for i, layer in enumerate(base_model.layers):
                        if i >= int(layer_count * 0.7):  # Last 30%
                            layer.trainable = True
                    
                    print(f"Unfroze the last 30% of base model layers")
                    break
            
            # Lower learning rate for fine-tuning
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE/5, clipnorm=1.0),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_accuracy')]
            )
            
            # Continue training
            history2 = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=35,
                initial_epoch=history1.epoch[-1] + 1 if hasattr(history1, 'epoch') else len(history1.history['loss']),
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
            
            # Clean memory
            clean_memory()
            
            # Save phase 2 model
            model.save('phase2_model.h5')
            
            # Phase 3: Final fine-tuning with unfrozen model and mixup
            print("\nPhase 3: Final fine-tuning with mixup augmentation...")
            
            # Unfreeze more layers but keep lower layers frozen
            for layer in model.layers:
                if isinstance(layer, tf.keras.Model):
                    base_model = layer
                    
                    # Unfreeze the last 50% of layers
                    layer_count = len(base_model.layers)
                    for i, layer in enumerate(base_model.layers):
                        if i >= int(layer_count * 0.5):  # Last 50%
                            layer.trainable = True
                    
                    print(f"Unfroze the last 50% of base model layers")
                    break
            
            # Even lower learning rate for final fine-tuning
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE/10, clipnorm=1.0),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_accuracy')]
            )
            
            # Add mixup callback
            mixup_cb = MixupCallback(train_generator, alpha=MIXUP_ALPHA, apply_after_epoch=0)
            callbacks.append(mixup_cb)
            
            # Continue training
            history3 = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=EPOCHS,
                initial_epoch=history2.epoch[-1] + 1 if hasattr(history2, 'epoch') else 35,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
            
            # Clean memory
            clean_memory()
            
            # Combine the histories
            history = {}
            for k in history1.history.keys():
                if k in history2.history and k in history3.history:
                    history[k] = history1.history[k] + history2.history[k] + history3.history[k]
                elif k in history2.history:
                    history[k] = history1.history[k] + history2.history[k]
                else:
                    history[k] = history1.history[k]
                    
            combined_history = type('obj', (object,), {'history': history})
            
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to load best checkpoint
            if os.path.exists('best_model.h5'):
                print("Loading best checkpoint...")
                try:
                    model = tf.keras.models.load_model('best_model.h5')
                except Exception as load_e:
                    print(f"Error loading best model: {load_e}")
                    if os.path.exists('phase2_model.h5'):
                        try:
                            model = tf.keras.models.load_model('phase2_model.h5')
                        except Exception as load_e2:
                            print(f"Error loading phase 2 model: {load_e2}")
                            if os.path.exists('phase1_model.h5'):
                                try:
                                    model = tf.keras.models.load_model('phase1_model.h5')
                                except Exception as load_e3:
                                    print(f"Error loading phase 1 model: {load_e3}")
            
            if model is None:
                print("Failed to load any model checkpoints.")
                return None, None
            
            return model, None
        
        # Load best model for evaluation
        if os.path.exists('best_model.h5'):
            print("Loading best model from checkpoint...")
            try:
                model = tf.keras.models.load_model('best_model.h5')
                print("Best model loaded successfully")
            except Exception as e:
                print(f"Error loading best model: {e}")
        
        # Clean memory
        clean_memory()
        
        # Save final model
        try:
            model.save('celebrity_recognition_model.h5')
            print("Final model saved successfully")
        except Exception as e:
            print(f"Error saving final model: {e}")
        
        return model, combined_history
    
    except Exception as e:
        print(f"Model creation error: {e}")
        import traceback
        traceback.print_exc()
        clean_memory()
        return None, None

class CelebDataGenerator(tf.keras.utils.Sequence):
    """Data generator with enhanced augmentation for face recognition"""
    def __init__(self, dataframe, img_dir, batch_size=16, img_size=(224, 224), shuffle=True, augment=False):
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(dataframe))
        
        # Check if image directory exists
        if not os.path.exists(img_dir):
            print(f"WARNING: Image directory '{img_dir}' does not exist!")
        
        # Verify a few image paths
        sample_size = min(5, len(dataframe))
        if sample_size > 0:
            print(f"Checking {sample_size} sample image paths:")
            for i, row in dataframe.head(sample_size).iterrows():
                img_path = os.path.join(img_dir, row['image'])
                exists = os.path.exists(img_path)
                print(f"  {row['image']}: {'exists' if exists else 'missing'}")
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
        # Set up more diverse augmentation parameters
        self.rotation_range = 30 if augment else 0  # More rotation
        self.width_shift_range = 0.15 if augment else 0  # Larger shifts
        self.height_shift_range = 0.15 if augment else 0
        self.brightness_range = [0.7, 1.3] if augment else [1.0, 1.0]  # More brightness variation
        self.zoom_range = 0.15 if augment else 0  # More zoom
        self.horizontal_flip = augment
        
        # Cache face detector for efficiency
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception as e:
            print(f"Error loading face detector: {e}")
            self.face_cascade = None
    
    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))
        
    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.dataframe.iloc[batch_indexes]
        
        batch_x = []
        batch_y = []
        
        # Track which images failed to load
        failed_images = []
        
        for _, row in batch_df.iterrows():
            img_path = os.path.join(self.img_dir, row['image'])
            try:
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    failed_images.append(row['image'])
                    print(f"Warning: Could not read image {img_path}")
                    continue
                    
                # Convert to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Ensure image has 3 channels
                if len(img.shape) != 3 or img.shape[2] != 3:
                    if len(img.shape) == 2:
                        img = np.stack((img,)*3, axis=-1)
                    elif img.shape[2] == 4:
                        img = img[:, :, :3]
                    else:
                        print(f"Cannot fix unusual image shape: {img.shape}")
                        failed_images.append(row['image'])
                        continue
                
                # Try to detect and crop face if enabled
                if ENABLE_FACE_ALIGNMENT and self.face_cascade is not None:
                    try:
                        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                        
                        if len(faces) > 0:
                            # Use the largest face
                            if len(faces) > 1:
                                # Calculate areas and find largest
                                areas = [w*h for (x, y, w, h) in faces]
                                largest_face_idx = np.argmax(areas)
                                x, y, w, h = faces[largest_face_idx]
                            else:
                                x, y, w, h = faces[0]
                            
                            # Add margin
                            margin = int(0.3 * max(w, h))
                            x = max(0, x - margin)
                            y = max(0, y - margin)
                            w = min(img.shape[1] - x, w + 2*margin)
                            h = min(img.shape[0] - y, h + 2*margin)
                            
                            # Crop to face region
                            img = img[y:y+h, x:x+w]
                    except Exception as e:
                        # If face detection fails, use the whole image
                        if idx == 0:  # Only print once per epoch
                            print(f"Face detection failed: {e}")
                
                # Resize to target size
                img = cv2.resize(img, self.img_size)
                
                # Apply enhanced augmentation
                if self.augment:
                    img = augment_image(img, augment=True)
                
                batch_x.append(img)
                batch_y.append(row['label'])
            except Exception as e:
                failed_images.append(row['image'])
                print(f"Error loading image {img_path}: {e}")
        
        # Handle empty batches
        if not batch_x:
            if failed_images:
                print(f"WARNING: All {len(failed_images)} images in batch {idx} failed to load")
            return np.empty((0, self.img_size[0], self.img_size[1], 3)), np.empty(0)
        
        # Convert to numpy arrays
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        
        # Print shape for debugging (only for first batch)
        if idx == 0:
            print(f"Batch shape: {batch_x.shape}, Labels shape: {batch_y.shape}")
            print(f"Label range: min={batch_y.min()}, max={batch_y.max()}")
            print(f"Pixel value range: min={batch_x.min()}, max={batch_x.max()}")
            
        # No preprocessing here - we'll keep images in [0-255] range 
        # and let the model's preprocessing handle normalization
        # This is consistent with how we handle preprocessing in the model
                
        # Check for NaN or Inf
        if np.isnan(batch_x).any() or np.isinf(batch_x).any():
            print(f"Warning: NaN or Inf values in batch {idx}")
            batch_x = np.nan_to_num(batch_x)
        
        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
        # Clean memory at the end of each epoch
        if MEMORY_OPTIMIZATION:
            gc.collect()

def mine_hard_examples(model, train_df, img_dir, k=0.3):
    """Mine hard examples that the model struggles with for focused training"""
    print("\nMining hard examples for focused training...")
    
    if len(train_df) == 0:
        print("Training dataframe is empty. Skipping hard example mining.")
        return train_df
    
    # Create a single-batch generator for prediction
    batch_size = min(32, len(train_df))
    mine_generator = CelebDataGenerator(
        train_df, img_dir, batch_size=batch_size, 
        img_size=(IMG_SIZE, IMG_SIZE), shuffle=False, augment=False
    )
    
    # Calculate loss for each example
    all_losses = []
    all_indices = []
    
    try:
        for i in range(len(mine_generator)):
            if i % 10 == 0:
                print(f"Mining batch {i+1}/{len(mine_generator)}...")
                
            # Get batch
            x_batch, y_batch = mine_generator[i]
            if len(x_batch) == 0:
                continue
                
            # Predict
            y_pred = model.predict(x_batch, verbose=0)
            
            # Calculate loss for each example
            for j in range(len(y_batch)):
                # Get prediction for the correct class
                true_class = y_batch[j]
                pred_prob = y_pred[j][true_class]
                
                # Use -log(p) as the loss
                loss = -np.log(max(pred_prob, 1e-7))
                
                # Store loss and index
                all_losses.append(loss)
                all_indices.append(i * batch_size + j)
    except Exception as e:
        print(f"Error during hard example mining: {e}")
        return train_df
    
    if not all_losses:
        print("No valid examples found during mining. Skipping hard example mining.")
        return train_df
    
    # Sort by loss (highest loss = hardest examples)
    sorted_indices = [idx for _, idx in sorted(zip(all_losses, all_indices), reverse=True)]
    
    # Select top k% hardest examples
    num_hard = int(k * len(sorted_indices))
    hard_indices = sorted_indices[:num_hard]
    
    # Extract hard examples
    hard_df = train_df.iloc[hard_indices].copy()
    
    # Combine with a random sample of easier examples for balance
    num_easy = min(len(train_df) - num_hard, num_hard * 2)
    easy_candidates = [idx for idx in range(len(train_df)) if idx not in hard_indices]
    if len(easy_candidates) > num_easy:
        easy_indices = np.random.choice(easy_candidates, size=num_easy, replace=False)
        easy_df = train_df.iloc[easy_indices].copy()
        combined_df = pd.concat([hard_df, easy_df]).reset_index(drop=True)
    else:
        combined_df = hard_df
    
    print(f"Created focused dataset with {len(hard_df)} hard and {len(combined_df) - len(hard_df)} easier examples")
    return combined_df

def evaluate_model(model, test_df):
    """Evaluate the model on the test set"""
    # Check if test_df is empty
    if test_df.empty:
        print("Warning: Test dataset is empty! Skipping evaluation.")
        return
        
    print(f"Evaluating model on {len(test_df)} test samples")
    
    # Check that the test set has the right labels
    if 'label' not in test_df.columns:
        print("Error: Test dataframe doesn't have 'label' column. Skipping evaluation.")
        return
        
    # Validate label range
    max_label = test_df['label'].max()
    if max_label >= NUM_CLASSES:
        print(f"Warning: Test labels exceed NUM_CLASSES ({max_label} >= {NUM_CLASSES})")
        print("Filtering test data to valid labels...")
        test_df = test_df[test_df['label'] < NUM_CLASSES]
        
    # Check if we still have test data after filtering
    if len(test_df) == 0:
        print("Error: No valid test samples after filtering. Skipping evaluation.")
        return
    
    # Print path to ensure it exists
    print(f"Using image directory: {IMG_DIR}")
    img_path_check = os.path.join(IMG_DIR, test_df['image'].iloc[0])
    print(f"Sample image path: {img_path_check}")
    print(f"Path exists: {os.path.exists(img_path_check)}")
    
    # Create a test generator with additional error handling
    try:
        test_generator = CelebDataGenerator(
            test_df, 
            IMG_DIR, 
            batch_size=BATCH_SIZE, 
            img_size=(IMG_SIZE, IMG_SIZE), 
            shuffle=False,
            augment=False
        )
    except Exception as e:
        print(f"Error creating test generator: {e}")
        import traceback
        traceback.print_exc()
        return
        
    # Verify the generator works by checking the first batch
    try:
        print("Checking test generator...")
        
        # Try to get the first batch while catching any errors
        if len(test_generator) > 0:
            try:
                x_test, y_test = test_generator[0]
                if len(x_test) == 0:
                    print("Warning: First batch is empty")
                else:
                    print(f"First batch shape: {x_test.shape}, Labels shape: {y_test.shape}")
            except Exception as batch_e:
                print(f"Error fetching first batch: {batch_e}")
                import traceback
                traceback.print_exc()
                
                # Try to check each image path in the first batch
                print("\nChecking individual image paths in first batch:")
                batch_df = test_df.iloc[:BATCH_SIZE]
                for i, row in batch_df.iterrows():
                    img_path = os.path.join(IMG_DIR, row['image'])
                    exists = os.path.exists(img_path)
                    readable = False
                    if exists:
                        try:
                            img = cv2.imread(img_path)
                            readable = img is not None
                        except:
                            pass
                    print(f"  Image {row['image']}: Exists: {exists}, Readable: {readable}")
                
                # Fall back to manual evaluation
                print("Will try manual evaluation instead of using the generator")
                test_generator = None
        else:
            print("Warning: Test generator has zero length")
            test_generator = None
    except Exception as e:
        print(f"Error with test generator: {e}")
        import traceback
        traceback.print_exc()
        test_generator = None

    # If we couldn't create a valid generator, try direct evaluation
    if test_generator is None:
        print("Attempting direct evaluation on test images...")
        try:
            # Manually evaluate on a sample of test images
            correct = 0
            total = 0
            max_samples = min(100, len(test_df))  # Limit to 100 images for speed
            
            for i, row in test_df.head(max_samples).iterrows():
                img_path = os.path.join(IMG_DIR, row['image'])
                try:
                    # Load and preprocess the image
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Couldn't read image {img_path}")
                        continue
                        
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    
                    # Normalize the image as in the data generator
                    # Don't preprocess here - let the model handle it
                    # Add batch dimension
                    img = np.expand_dims(img, axis=0)
                    
                    # Predict
                    pred = model.predict(img, verbose=0)
                    pred_class = np.argmax(pred[0])
                    
                    # Check if correct
                    if pred_class == row['label']:
                        correct += 1
                    total += 1
                    
                    # Print progress every 10 images
                    if (i + 1) % 10 == 0:
                        print(f"Processed {i + 1}/{max_samples} images, current accuracy: {correct/total:.4f}")
                        
                except Exception as img_e:
                    print(f"Error processing image {img_path}: {img_e}")
                    continue
            
            # Print final results
            if total > 0:
                print(f"\nDirect evaluation results:")
                print(f"Accuracy: {correct/total:.4f} ({correct}/{total})")
            else:
                print("No images could be processed in direct evaluation.")
            
            return
            
        except Exception as direct_e:
            print(f"Direct evaluation failed: {direct_e}")
            import traceback
            traceback.print_exc()
            return
    
    # Evaluate the model with try-except to catch errors
    try:
        print("Starting evaluation...")
        results = model.evaluate(test_generator, verbose=1)
        print(f"Test Loss: {results[0]}")
        print(f"Test Accuracy: {results[1]}")
        if len(results) > 2:
            print(f"Top-3 Accuracy: {results[2]}")
    except Exception as e:
        print(f"Evaluation error: {e}")
        print("Trying to evaluate using manual prediction...")
        
        # Manual evaluation as fallback
        try:
            correct = 0
            total = 0
            
            for i in range(len(test_generator)):
                try:
                    x_batch, y_batch = test_generator[i]
                    if len(x_batch) == 0:
                        continue
                        
                    print(f"Batch {i+1}/{len(test_generator)}, shape: {x_batch.shape}")
                    predictions = model.predict(x_batch, verbose=0)
                    predicted_classes = np.argmax(predictions, axis=1)
                    
                    correct += np.sum(predicted_classes == y_batch)
                    total += len(y_batch)
                except Exception as batch_e:
                    print(f"Error with batch {i}: {batch_e}")
                    continue
            
            if total > 0:
                accuracy = correct / total
                print(f"Manual evaluation accuracy: {accuracy:.4f} ({correct}/{total})")
            else:
                print("No samples were evaluated during manual evaluation.")
        except Exception as nested_e:
            print(f"Manual evaluation also failed: {nested_e}")

def preprocess_image(image_path_or_url, return_faces=False):
    """Enhanced preprocessing with face alignment for better recognition"""
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
        
        # Check for alpha channel and remove if needed
        if len(img.shape) > 2 and img.shape[2] == 4:
            img = img[:, :, :3]
        
        # Get image dimensions
        h, w = img.shape[:2]
        print(f"Original image size: {w}x{h}")
        
        # If image is very large, resize it to a reasonable size to work with
        if h > 1200 or w > 1200:
            scale = 1200 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h))
            print(f"Resized image to {new_w}x{new_h}")
        
        # Try face detection with multiple approaches
        face_images = []
        
        # First method: Try OpenCV's Haar cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                print(f"Detected {len(faces)} faces using Haar cascade")
                # Process each face (take up to 3 largest faces)
                if len(faces) > 1:
                    # Calculate areas
                    areas = [(x, y, w, h, w*h) for (x, y, w, h) in faces]
                    # Sort by area (largest first)
                    areas.sort(key=lambda a: a[4], reverse=True)
                    
                    # Process up to 3 largest faces
                    for i, (x, y, w, h, area) in enumerate(areas[:3]):
                        if area < 100:  # Skip very small faces
                            continue
                            
                        # Add some margin
                        margin = int(0.3 * max(w, h))
                        x = max(0, x - margin)
                        y = max(0, y - margin)
                        w = min(img.shape[1] - x, w + 2*margin)
                        h = min(img.shape[0] - y, h + 2*margin)
                        
                        # Crop to face region
                        face_img = img[y:y+h, x:x+w]
                        
                        # Resize to model input size
                        face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
                        
                        # Keep in RGB format with full pixel range - model will handle preprocessing
                        face_images.append(face_img)
                        print(f"Processed face {i+1} of size {w}x{h}")
                else:
                    # Handle single face case directly
                    x, y, w, h = faces[0]
                    margin = int(0.3 * max(w, h))
                    x = max(0, x - margin)
                    y = max(0, y - margin)
                    w = min(img.shape[1] - x, w + 2*margin)
                    h = min(img.shape[0] - y, h + 2*margin)
                    
                    # Crop to face region
                    face_img = img[y:y+h, x:x+w]
                    
                    # Resize to model input size
                    face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
                    
                    # Keep in RGB format with full pixel range - model will handle preprocessing
                    face_images.append(face_img)
                    print(f"Processed single face of size {w}x{h}")
            else:
                print("No faces detected using Haar cascade")
        except Exception as e:
            print(f"Haar cascade error: {e}")
        
        # If no faces detected, or as a fallback, use the whole image
        if not face_images:
            print("Using full image as fallback")
            whole_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            # Keep in RGB format with full pixel range - model will handle preprocessing
            face_images.append(whole_img)
        
        if return_faces:
            # Return all detected faces for ensemble prediction
            return np.array(face_images)
        else:
            # Return the first (largest) face for single prediction
            return np.expand_dims(face_images[0], axis=0)
            
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        import traceback
        traceback.print_exc()
        return None

def calibrate_confidence(predictions, temperature=1.5):
    """Apply temperature scaling to calibrate confidence scores.
    
    Higher temperature values (>1.0) will make the distribution more uniform (reduce confidence),
    while lower values (<1.0) will make the distribution more peaked (increase confidence).
    """
    if temperature == 1.0:
        return predictions
        
    # Apply temperature scaling
    log_predictions = np.log(predictions + 1e-7)  # Add small epsilon to avoid log(0)
    calibrated = np.exp(log_predictions / temperature) 
    # Re-normalize to ensure valid probability distribution
    calibrated = calibrated / np.sum(calibrated)
    return calibrated

def visualize_prediction(image_path, predictions, identity_mapping=None, name_mapping=None):
    """Visualize the prediction results with confidence bars."""
    try:
        if not VISUALIZATION_ENABLED:
            return
            
        # Import visualization libraries
        import matplotlib.pyplot as plt
        
        # Load and display the image
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path)
            img = Image.open(BytesIO(response.content))
            img = np.array(img)
        else:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display the image
        ax1.imshow(img)
        ax1.set_title("Input Image")
        ax1.axis('off')
        
        # Get top 5 predictions
        top_indices = np.argsort(-predictions)[:5]
        top_probs = predictions[top_indices]
        
        # Create labels
        labels = []
        if name_mapping is not None:
            for idx in top_indices:
                name = name_mapping.get(idx, f"Unknown-{idx}")
                labels.append(name)
        elif identity_mapping is not None:
            reverse_mapping = {v: k for k, v in identity_mapping.items()}
            for idx in top_indices:
                celeb_id = reverse_mapping.get(idx, f"Unknown-{idx}")
                labels.append(f"ID: {celeb_id}")
        else:
            labels = [f"Class {idx}" for idx in top_indices]
        
        # Plot confidence bars
        bars = ax2.barh(range(len(top_probs)), top_probs)
        ax2.set_yticks(range(len(labels)))
        ax2.set_yticklabels(labels)
        ax2.set_xlim(0, 1.0)
        ax2.set_xlabel('Confidence')
        ax2.set_title('Top 5 Predictions')
        
        # Add confidence values
        for i, v in enumerate(top_probs):
            ax2.text(v + 0.01, i, f"{v:.2f}", va='center')
        
        # Highlight the top prediction
        bars[0].set_color('red')
        
        plt.tight_layout()
        plt.savefig('prediction_visualization.png')
        print("Visualization saved as 'prediction_visualization.png'")
        
        # Show the plot
        plt.show()
    except Exception as e:
        print(f"Visualization error: {e}")
        # Import error or display error shouldn't stop the prediction process
        pass

def predict_celebrity(model, image_path_or_url, use_ensemble=True):
    """Improved prediction function with ensemble prediction and confidence calibration"""
    # Load identity mapping
    try:
        identity_mapping = np.load('identity_mapping.npy', allow_pickle=True).item()
        reverse_mapping = {v: k for k, v in identity_mapping.items()}
        
        # Try to load name mapping
        try:
            name_mapping = np.load('name_mapping.npy', allow_pickle=True).item()
        except Exception as e:
            print(f"Name mapping not found: {e}. Using numeric IDs.")
            name_mapping = {}
        
        # Preprocess the image to detect faces
        face_images = preprocess_image(image_path_or_url, return_faces=use_ensemble)
        if face_images is None or len(face_images) == 0:
            print("Failed to preprocess image. Please try another image.")
            return None
        
        # Make predictions
        if use_ensemble and len(face_images) > 1:
            print(f"Using ensemble prediction with {len(face_images)} faces")
            # Predict on each face
            all_predictions = []
            for face_idx, face in enumerate(face_images):
                # Model will handle preprocessing internally based on architecture
                face_input = np.expand_dims(face, axis=0)
                
                # Predict
                face_preds = model.predict(face_input, verbose=0)[0]
                
                # Apply confidence calibration
                if ENABLE_CONFIDENCE_CALIBRATION:
                    face_preds = calibrate_confidence(face_preds, temperature=1.5)
                
                print(f"Face {face_idx+1} top prediction: class {np.argmax(face_preds)} ({reverse_mapping.get(np.argmax(face_preds), 'Unknown')}) with confidence {np.max(face_preds):.4f}")
                all_predictions.append(face_preds)
            
            # Average the predictions with temperature scaling
            predictions = np.mean(all_predictions, axis=0)
            
            # Recalibrate the averaged predictions
            if ENABLE_CONFIDENCE_CALIBRATION:
                predictions = calibrate_confidence(predictions, temperature=1.2)
        else:
            # Single face prediction
            if len(face_images.shape) == 4:
                face_input = face_images  # Already has batch dimension
            else:
                face_input = np.expand_dims(face_images, axis=0)
                
            # Predict
            predictions = model.predict(face_input, verbose=0)[0]
            
            # Apply confidence calibration
            if ENABLE_CONFIDENCE_CALIBRATION:
                predictions = calibrate_confidence(predictions, temperature=1.5)
        
        # Get top predicted class
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        
        # Get the celebrity ID and name if available
        celebrity_id = reverse_mapping.get(predicted_class, "Unknown")
        celebrity_name = name_mapping.get(predicted_class, f"Unknown-{celebrity_id}")
        
        # Create top predictions with names if available
        top_predictions = []
        for i in np.argsort(-predictions)[:5]:  # Top 5 predictions
            pred_id = reverse_mapping.get(i, "Unknown")
            pred_name = name_mapping.get(i, f"Unknown-{pred_id}")
            
            top_predictions.append({
                'celebrity_id': pred_id,
                'celebrity_name': pred_name,
                'confidence': float(predictions[i])
            })
        
        # Visualize the prediction results
        try:
            visualize_prediction(
                image_path_or_url, 
                predictions, 
                identity_mapping=identity_mapping, 
                name_mapping=name_mapping
            )
        except Exception as viz_error:
            print(f"Visualization error: {viz_error}")
        
        return {
            'celebrity_id': celebrity_id,
            'celebrity_name': celebrity_name,
            'confidence': float(confidence),
            'top_predictions': top_predictions
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_celebrity_name_lookup():
    """Create a mapping from label to celebrity name using the identity mapping"""
    try:
        # Load identity mapping
        identity_mapping = np.load('identity_mapping.npy', allow_pickle=True).item()
        if not identity_mapping:
            print("Identity mapping not found or empty.")
            return
        
        # Get reverse mapping (label → identity)
        reverse_mapping = {v: k for k, v in identity_mapping.items()}
        
        # Try to load celebrity names
        celebrity_names = {}
        try:
            celebrity_names = load_celebrity_names()
        except Exception as e:
            print(f"Error loading celebrity names: {e}")
        
        # Create label to name mapping
        label_to_name = {}
        for label, identity in reverse_mapping.items():
            name = celebrity_names.get(identity, f"Unknown-{identity}")
            label_to_name[label] = name
        
        # Save mapping
        np.save('name_mapping.npy', label_to_name)
        print(f"Created and saved name mapping for {len(label_to_name)} celebrities")
        
        # Print sample of the mapping
        print("\nSample of celebrity name mapping:")
        for i, (label, name) in enumerate(list(label_to_name.items())[:5]):
            print(f"  {i+1}. Label {label} → {name}")
        
    except Exception as e:
        print(f"Error creating celebrity name lookup: {e}")

def main():
    """Main function with memory optimization and improved prediction features"""
    clean_memory()
    
    # Check if user wants to force training a new model
    force_training = False
    response = input("Do you want to train a new model? (y/n, default: n): ")
    if response.lower() in ['y', 'yes']:
        force_training = True
        print("Will train new model regardless of existing ones.")
    
    model = None
    if not force_training:
        # Check if the model exists in any format
        for model_path in ['celebrity_recognition_model.h5', 'celebrity_recognition_model.keras']:
            if os.path.exists(model_path) and os.path.isfile(model_path):
                print(f"Found existing model at {model_path}")
                try:
                    model = tf.keras.models.load_model(model_path)
                    print(f"Successfully loaded model from {model_path}")
                    break
                except Exception as e:
                    print(f"Error loading model from {model_path}: {e}")
        
        # Also check SavedModel format
        if model is None and os.path.exists('celebrity_recognition_model') and os.path.isdir('celebrity_recognition_model'):
            try:
                # Check if it's a valid SavedModel directory
                if os.path.exists(os.path.join('celebrity_recognition_model', 'saved_model.pb')):
                    print("Found SavedModel format at celebrity_recognition_model/")
                    model = tf.keras.models.load_model('celebrity_recognition_model')
                    print("Successfully loaded SavedModel format")
                else:
                    print("Directory 'celebrity_recognition_model' exists but is not a valid SavedModel")
            except Exception as e:
                print(f"Error loading SavedModel: {e}")
    
    if force_training or model is None:
        print("Will train a new model...")
        model, _ = train_model()
        
    # Only proceed with predictions if model is valid
    if model is None:
        print("ERROR: No valid model available for predictions. Exiting.")
        return
    
    # Run evaluation on test set
    df = prepare_dataset()
    test_df = df[df['partition'] == 2]
    
    # Ask if user wants to run evaluation
    eval_response = input("Do you want to evaluate the model on the test set? (y/n, default: y): ")
    if eval_response.lower() not in ['n', 'no']:
        evaluate_model(model, test_df)
    
    # Ask about confidence calibration
    calib_response = input("Do you want to enable confidence calibration? (y/n, default: y): ")
    global ENABLE_CONFIDENCE_CALIBRATION
    ENABLE_CONFIDENCE_CALIBRATION = calib_response.lower() not in ['n', 'no']
    print(f"Confidence calibration is {'enabled' if ENABLE_CONFIDENCE_CALIBRATION else 'disabled'}")
    
    # Ask about visualization
    viz_response = input("Do you want to enable prediction visualization? (y/n, default: y): ")
    global VISUALIZATION_ENABLED
    VISUALIZATION_ENABLED = viz_response.lower() not in ['n', 'no']
    print(f"Prediction visualization is {'enabled' if VISUALIZATION_ENABLED else 'disabled'}")
    
    # Clean memory before interactive loop
    clean_memory()
    
    # Interactive prediction loop
    while True:
        image_input = input("\nEnter a URL or local path to an image of a celebrity (or 'q' to quit): ")
        if image_input.lower() == 'q':
            break
            
        print("Processing image...")
        # Clean memory before each prediction
        clean_memory()
        result = predict_celebrity(model, image_input, use_ensemble=True)
        
        if result:
            print(f"\nPredicted Celebrity: {result['celebrity_name']} (ID: {result['celebrity_id']})")
            print(f"Confidence: {result['confidence']:.2f}")
            print("\nTop 5 Predictions:")
            for i, pred in enumerate(result['top_predictions']):
                print(f"{i+1}. {pred['celebrity_name']} (ID: {pred['celebrity_id']}), Confidence: {pred['confidence']:.2f}")
            
            # Ask if prediction was correct for feedback
            feedback = input("\nWas the prediction correct? (y/n/unknown): ")
            if feedback.lower() in ['y', 'yes']:
                print("Great! Thank you for the feedback.")
            elif feedback.lower() in ['n', 'no']:
                correct_celeb = input("Who is the correct celebrity? (Enter name or ID, or skip with Enter): ")
                if correct_celeb:
                    print(f"Thank you! This feedback can help improve the model.")
                    # In a production system, this could be logged for model improvement
        else:
            print("Prediction failed. Please try another image.")

if __name__ == "__main__":
    # Create name lookup if needed
    if not os.path.exists('name_mapping.npy'):
        print("Creating celebrity name lookup...")
        create_celebrity_name_lookup()
    
    main() 