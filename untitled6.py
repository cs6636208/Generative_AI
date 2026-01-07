import tensorflow as tf
print(tf.__version__)

# Cell 2
import os, sys, glob, math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

import faiss

print("TensorFlow:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# Cell 3
IMG_SIZE = (224, 224)   # ขนาด input
EMBED_DIM = 2048        # ResNet50V2 global pool output

def create_embedding_model():
    base = ResNet50V2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    model = tf.keras.Model(inputs=base.input, outputs=x)
    return model

model = create_embedding_model()
model.summary()

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Cell 4
def load_and_preprocess(path):
    img = Image.open(path).convert('RGB')
    img = img.resize(IMG_SIZE)
    arr = img_to_array(img)
    arr = preprocess_input(arr)   # ใช้ preprocessing ของ ResNetV2
    return arr

def extract_features_from_paths(paths, batch_size=32):
    features = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        batch_imgs = np.stack([load_and_preprocess(p) for p in batch_paths], axis=0)
        emb = model.predict(batch_imgs, verbose=0)
        features.append(emb.astype('float32'))
    features = np.vstack(features)
    return features

# Cell 5 — ปรับ path dataset ให้ตรงกับที่คุณเก็บ
DATASET_ROOT = "/content/drive/MyDrive/ProjectML-@LAS/ProjectML-@LAS/Google Images"  # เปลี่ยนเป็น path ของคุณ
# ตัวอย่างโครงสร้าง: /content/dataset/train/<class>/*.jpg

all_image_paths = []
for split in ["train", "val", "test"]:
    folder = os.path.join(DATASET_ROOT, split)
    if not os.path.exists(folder):
        continue
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".jpg",".jpeg",".png")):
                all_image_paths.append(os.path.join(root, f))

print("Total images found:", len(all_image_paths))

# สร้าง features (อาจกินเวลา ขึ้นกับจำนวนภาพ และใช้ GPU เร็วขึ้น)
BATCH_SIZE = 32
features = extract_features_from_paths(all_image_paths, batch_size=BATCH_SIZE)
print("features shape:", features.shape)

# Normalize features (สำหรับใช้ cosine similarity ผ่าน inner product)
# วิธี: L2-normalize แล้วใช้ IndexFlatIP (inner product ≈ cosine)
def l2_normalize(x, axis=1, eps=1e-10):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)

features_norm = l2_normalize(features)

# save to disk
np.save("image_paths.npy", np.array(all_image_paths))
np.save("features.npy", features_norm)
print("Saved features and paths.")