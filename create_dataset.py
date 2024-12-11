import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# load csv of moon data
df = pd.read_csv('tensorflow-moons/train/_annotations.csv')
base_dir = 'tensorflow-moons/train/'

# load and preprocess image to 224x224
def load_image_and_label(filename):
    path = base_dir + filename
    img = Image.open(path).convert('L')  # Convert to grayscale
    img = img.resize((224, 224))  # Resize to 224x224
    return np.array(img), df[df['filename'] == filename]['class'].iloc[0]

# process all images and create labels
x_img = []
y_label = []
for _, row in df.iterrows():
    img, label = load_image_and_label(row['filename'])
    x_img.append(img)
    y_label.append(label)
    
    # DEBUG
    print('X: ',  x_img)
    print('y: ',  y_label)

# convert lists to arrays
X = np.array(x_img)
y = np.array(y_label)