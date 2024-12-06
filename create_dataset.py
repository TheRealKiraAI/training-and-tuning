import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

def parse_json_annotation(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    label_categories = data['categories']
    images = data['images']
    annotations_data = data['annotations']

    # maps the moon phases as labels
    category_map = {cat['id']: cat['name'] for cat in label_categories}

    # parse the annotations
    annotations = []
    for annotation in annotations_data:
        id = annotation['id']
        image_id = annotation['image_id']
        category_id = annotation['category_id']

        # DEBUG
        print(f"Processing category: {category_id}, ID: {image_id}")

        if category_id is not None:
            category_name = category_map[category_id]
            print("Category Name: ", category_name)
            obj = {'cat name': category_name, 'id': category_id}
            annotations.append({
                'image_id': image_id,
                'category_id': category_id,
                # 'xmin': 0,
                # 'ymin': 0,
                'class': obj
            })
        else: print(f"CATEGORY NONE")

    return annotations

def main():
    # Directory containing JSON annotations
    annotation_dir = 'moons/json/train/'
    
    # List to store all parsed annotations
    annotations = []
    
    # Parse JSON annotations
    for json_file in os.listdir(annotation_dir):
        if json_file.endswith('.json'):
            json_path = os.path.join(annotation_dir, json_file)
            annotations.extend(parse_json_annotation(json_path))
    
    # Prepare data
    # X = []
    # y = []
    # for ann in annotations:
    #     img_path = os.path.join(annotation_dir, ann['filename'])
    #     img = Image.open(img_path)
    #     img_array = np.array(img.resize((224, 224)))
    #     img_array = img_array / 255.0  # Normalize pixel values
        
    #     # Create a binary mask for the entire image
    #     mask = np.ones((224, 224), dtype=np.uint8)
        
    #     # Combine image and mask
    #     combined = np.dstack((img_array, mask))
        
    #     X.append(combined)
    #     y.append(ann['class']['name'])

    # print(f"Number of annotations: {len(annotations)}")
    # print(f"Number of images (X): {len(X)}")
    # print(f"Number of labels (y): {len(y)}")

    # # Create labels based on class names
    # label_mapping = {
    #     'New Moon': 0,
    #     'First Quarter Moon': 1,
    #     'Full Moon': 2,
    #     'Third Quarter Moon': 3,
    #     'Waning Gibbous Moon': 4,
    #     'Wanning Crescent Moon': 5,
    #     'Waxing Crescent Moon': 6,
    #     'Waxing Gibbous Moon': 7
    # }
    # print("Contents of y before filtering:", y)
    
    # # Handle potential missing labels
    # valid_labels = set(label_mapping.keys())
    # valid_indices = [i for i, label in enumerate(y) if label in valid_labels]
    
    # X = [X[i] for i in valid_indices]
    # y = [y[i] for i in valid_indices]

    # # Debugging: Print the number of valid images and labels after filtering
    # print(f"Number of valid images (X) after filtering: {len(X)}")
    # print(f"Number of valid labels (y) after filtering: {len(y)}")

    # labelled_data = [(np.array(x), label_mapping[y]) for x, y in zip(X, y)]
    # print(labelled_data)
    # print([(np.array(x), label_mapping[y]) for x, y in zip(X, y)])
    # # Split data into training, validation, and test sets
    # train_data, val_test_data = train_test_split(labelled_data, test_size=0.3, random_state=42)
    # val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=42)

    # # Save training data
    # train_dir = 'moons/json/train/'
    # os.makedirs(train_dir, exist_ok=True)
    # for i, (data, label) in enumerate(train_data):
    #     img = Image.fromarray((data[..., :3] * 255).astype(np.uint8))
    #     mask = (data[..., 3] * 255).astype(np.uint8)
        
    #     img.save(os.path.join(train_dir, f'train_{i}_label_{label}.jpg'))
    #     np.save(os.path.join(train_dir, f'train_{i}_mask.npy'), mask)

    # # Save validation data
    # val_dir = 'path/to/validation'
    # os.makedirs(val_dir, exist_ok=True)
    # for i, (data, label) in enumerate(val_data):
    #     img = Image.fromarray((data[..., :3] * 255).astype(np.uint8))
    #     mask = (data[..., 3] * 255).astype(np.uint8)
        
    #     img.save(os.path.join(val_dir, f'validation_{i}_label_{label}.jpg'))
    #     np.save(os.path.join(val_dir, f'validation_{i}_mask.npy'), mask)

    # # Save test data
    # test_dir = 'path/to/test'
    # os.makedirs(test_dir, exist_ok=True)
    # for i, (data, label) in enumerate(test_data):
    #     img = Image.fromarray((data[..., :3] * 255).astype(np.uint8))
    #     mask = (data[..., 3] * 255).astype(np.uint8)
        
    #     img.save(os.path.join(test_dir, f'test_{i}_label_{label}.jpg'))
    #     np.save(os.path.join(test_dir, f'test_{i}_mask.npy'), mask)

    # # Create CSV file mapping image filenames to labels
    # import csv
    # with open('moon_phase_labels.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['filename', 'label'])
    #     for filename, label in [('train_0_label_0.jpg', 0), ('train_1_label_1.jpg', 1),
    #                              ('train_2_label_2.jpg', 2), ('train_3_label_3.jpg', 3),
    #                              ('validation_0_label_0.jpg', 0), ('validation_1_label_1.jpg', 1),
    #                              ('validation_2_label_2.jpg', 2), ('validation_3_label_3.jpg', 3),
    #                              ('test_0_label_0.jpg', 0), ('test_1_label_1.jpg', 1),
    #                              ('test_2_label_2.jpg', 2), ('test_3_label_3.jpg', 3)]:
    #         writer.writerow([filename, label])

if __name__ == "__main__":
    main()
