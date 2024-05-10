import streamlit as st
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
import os


def run():
    st.title('Food Classification Prediction')

    # Melakukan loading pickle files
    classifier_model = load_model('food_cnn_model1.keras')

    st.header('Food Photo Uploader')

    uploaded_file = st.file_uploader("Upload foto makananmu!", 
                         type=['png','jpg','jpeg'], 
                         accept_multiple_files=False)

    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image)
        

        # Make predictions
        if st.button('Predict'):
            result, idx = predict_class(image, classifier_model)
            st.write(result)

            data_kalori = [576, 52, 322, 89, 43, 353, 190, 361, 120, 361, 247, 247, 34, 717, 
                           41, 402, 239, 312, 120, 70, 406, 60, 230, 155, 9, 31, 413, 145, 120, 304,
                           260, 23, 53, 680, 607, 15, 25, 40, 420, 158, 12, 266, 87, 130, 15, 336, 208,
                           100, 330, 32, 31, 0, 0, 18, 82, 0, 0, 1, 85, 82, 17]
            st.write('Jumlah kalori : {}'.format(data_kalori[idx]))

def predict_class(image, classifier_model):
    test_image = image.resize((260, 260))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image/255.0
    test_image = np.expand_dims(test_image, axis = 0)
    test_image_fin = np.vstack([test_image])
    class_names = ['almonds','apple','avocado','banana','beer','biscuits','boisson-au-glucose-50g','bread-french-white-flour','bread-sourdough',
                   'bread-white','bread-whole-wheat','bread-wholemeal','broccoli','butter','carrot','cheese','chicken','chips-french-fries','coffee-with-caffeine','corn','croissant','cucumber','dark-chocolate',
                   'egg','espresso-with-caffeine','french-beans','gruyere','ham-raw','hard-cheese','honey','jam','leaf-spinach','mandarine','mayonnaise','mixed-nuts',
                   'mixed-salad-chopped-without-sauce','mixed-vegetables','onion','parmesan','pasta-spaghetti','pickle','pizza-margherita-baked','potatoes-steamed','rice',
                   'salad-leaf-salad-green','salami','salmon','sauce-savoury','soft-cheese','strawberries','sweet-pepper','tea','tea-green','tomato','tomato-sauce',
                   'water','water-mineral','white-coffee-with-caffeine','wine-red','wine-white','zucchini']
    predictions = classifier_model.predict(test_image)
    idx = np.argmax(predictions)
    image_class = class_names[idx]
    result = "Model memprediksi gambar yang di-upload adalah : {}".format(image_class)
    return result, idx