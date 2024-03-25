import pandas as pd 
import numpy as np 
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
import tensorflow 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

model = ResNet50 (weights = "imagenet", include_top = False, input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([model, 
                                     GlobalMaxPooling2D()])

model.summary()

def extract_feature(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))
    img_arr = image.img_to_array(img)
    expanded_img_arr = np.expand_dims(img_arr, axis=0)
    preprocessed_img = preprocess_input(expanded_img_arr)
    res = model.predict(preprocessed_img).flatten()
    normalized_res = res / norm(res)
    
    return normalized_res

filenames = []
for file in os.listdir("images1"):
    filenames.append(os.path.join("images1", file))
print(len(filenames))
print(filenames[0:5])

feature_list = []
for file in tqdm(filenames):
    feature_list.append(extract_feature(file, model))

pickle.dump(feature_list, open("embeddings.pkl", "wb"))
pickle.dump(filenames, open("filenames.pkl", "wb"))
