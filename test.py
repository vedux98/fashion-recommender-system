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



feature_list = np.array(pickle.load(open("embeddings.pkl", "rb")))
print(feature_list)
filenames = pickle.load(open("filenames.pkl","rb"))

model = ResNet50 (weights = "imagenet", include_top = False, input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([model, 
                                     GlobalMaxPooling2D()])


img = image.load_img("sample/th-2299212019.jpg", target_size=(224,224))
img_arr = image.img_to_array(img)
expanded_img_arr = np.expand_dims(img_arr, axis=0)
preprocessed_img = preprocess_input(expanded_img_arr)
res = model.predict(preprocessed_img).flatten()
normalized_res1 = res / norm(res)
    
from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors = 5, algorithm = "brute", metric= "euclidean")
neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors([normalized_res1])
print("indices")
for file in indices[0]:
    print(filenames[file])
