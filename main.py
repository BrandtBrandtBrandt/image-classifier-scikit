import os
import pickle

from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Prepare data
img2vec = Img2Vec()

data_dir = './data/intel_image_classification-dataset'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

data = {}
for j, dir_ in enumerate([train_dir, val_dir]):
    features = []
    labels = []
    for category in os.listdir(dir_):
        category_path = os.path.join(dir_, category)
        if not os.path.isdir(category_path):  # Skip if not a directory
            continue
        for img_name in os.listdir(category_path):
            if img_name == '.DS_Store':  # Skip .DS_Store files
                continue
            img_path = os.path.join(category_path, img_name)
            if not os.path.isfile(img_path):  # Skip if not a file
                continue
            try:
                img = Image.open(img_path).convert('RGB')  # Ensure image is in RGB mode
                img_features = img2vec.get_vec(img)
                features.append(img_features)
                labels.append(category)
            except Exception as e:
                print(f"Error processing file {img_path}: {e}")
                continue
    data_key = 'training_data' if j == 0 else 'validation_data'
    label_key = 'training_labels' if j == 0 else 'validation_labels'
    data[data_key] = features
    data[label_key] = labels

# Train model
model = RandomForestClassifier(random_state=0)
model.fit(data['training_data'], data['training_labels'])

# Test performance
y_pred = model.predict(data['validation_data'])
score = accuracy_score(y_pred, data['validation_labels'])

print(score)

# Save the model
with open('./model.p', 'wb') as f:
    pickle.dump(model, f)
