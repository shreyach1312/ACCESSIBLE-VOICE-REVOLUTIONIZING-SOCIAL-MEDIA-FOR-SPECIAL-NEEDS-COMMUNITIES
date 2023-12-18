import json
import tensorflow as tf
from keras.models import load_model
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.sequence import pad_sequences
import collections
from keras.preprocessing import image
from keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
from keras.models import Model



# Read the files word_to_idx.pkl and idx_to_word.pkl to get the mappings between word and index
word_to_index = {}
with open ("Data/textFiles/word_to_idx.pkl", 'rb') as file:
    word_to_index = pd.read_pickle(file)

index_to_word = {}
with open ("Data/textFiles/idx_to_word.pkl", 'rb') as file:
    index_to_word = pd.read_pickle(file)



print("Loading the model...")
model = load_model('model816_19.h5')

resnet50_model = ResNet50 (weights = 'imagenet', input_shape = (224, 224, 3))
resnet50_model = Model (resnet50_model.input, resnet50_model.layers[-2].output)



# Generate Captions for a random image in test dataset
def predict_caption(photo):

    inp_text = "startseq"

    for i in range(38):
        sequence = [word_to_index[w] for w in inp_text.split() if w in word_to_index]
        sequence = pad_sequences([sequence], maxlen=38, padding='post')

        ypred = model.predict([photo, sequence])
        ypred = ypred.argmax()
        word = index_to_word[ypred]

        inp_text += (' ' + word)

        if word == 'endseq':
            break

    final_caption = inp_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption



def preprocess_image (img):
    img = tf.keras.utils.load_img(img, target_size=(224, 224))
    img = tf.keras.utils.img_to_array(img)

    # Convert 3D tensor to a 4D tendor
    img = np.expand_dims(img, axis=0)

    #Normalize image accoring to ResNet50 requirement
    img = preprocess_input(img)

    return img


# A wrapper function, which inputs an image and returns its encoding (feature vector)
def encode_image (img):
    img = preprocess_image(img)

    feature_vector = resnet50_model.predict(img)
    # feature_vector = feature_vector.reshape((-1,))
    return feature_vector


def generate(photo):
  print("Encoding the image ...")
  
  photo = encode_image(photo).reshape((1, 2048))



  print("Running model to generate the caption...")
  caption = predict_caption(photo)


  print(caption)
  return caption



import streamlit as st
from PIL import Image
from io import StringIO
import pandas as pd
#from generate1 import generate

def main():
    image = Image.open("static/priv2.png")
    logo = Image.open("static/icon.png")
    st.set_page_config(page_title="Image Caption Generator", page_icon=logo)

    st.image(image)
    st.title("Neural image caption generator")

    st.sidebar.title("Neural image caption generator")


    
    uploaded_file = st.file_uploader("Choose a file",type=['jpg'])
    if uploaded_file is not None:
        
        caption= generate(uploaded_file)
        with st.expander("see output"): 
            st.image(uploaded_file, caption)
if __name__ == "__main__":
    main()