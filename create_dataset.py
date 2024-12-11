import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# load csv of moon data
df = pd.read_csv('tensorflow-moons/train/_annotations.csv')
base_dir = 'tensorflow-moons/train/'

class_mapping = {label: idx for idx, label in enumerate(df['class'].unique())}

# load and preprocess image to 224x224
def load_image_and_label(filename):
    full_path = base_dir + filename
    img = Image.open(full_path).convert('L')  # Convert to grayscale
    img = img.resize((224, 224))  # Resize to 224x224
    label_str = df[df['filename'] == filename]['class'].iloc[0]
    label_int = class_mapping[label_str]  # Convert string label to integer
    return np.array(img), label_int

# process all images and create labels
x_img = []
y_label = []
for _, row in df.iterrows():
    img, label = load_image_and_label(row['filename'])
    x_img.append(img)
    y_label.append(label)
    
    # DEBUG
    # print('X: ',  x_img)
    # print('y: ',  y_label)

# convert lists to arrays
X = np.array(x_img)
y = np.array(y_label)

# one-hot encode labels
# num_classes = len(set(df['class']))
# y = to_categorical(y, num_classes=num_classes)
num_classes = len(class_mapping)
y = to_categorical(y, num_classes=num_classes)

# split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")

# save preprocessed data
np.savez('moon_phase_data.npz', X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
