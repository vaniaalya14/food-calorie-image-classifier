import streamlit as st
import pandas as pd
import pickle as pkl
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def run():
    st.title('Exploratory Data Analysis Food Classification')
    
    # Menampilkan gambar
    title_pict = Image.open('food_image.jpg')
    st.image(title_pict, caption='Food',use_column_width='always')

    # Melakukan loading table data
    @st.cache_data
    def fetch_data():
        df = pd.read_csv('food_classif.csv')
        return df

    df = fetch_data()

    # Contoh data
    st.subheader("Image Samples")
    train_files = glob.glob("foodImage/*")
    fig, ax = plt.subplots(ncols=5, figsize=(30,10))
    for i in range(5):
        train_image = plt.imread(train_files[i])
        ax[i].imshow(train_image)
    st.pyplot(fig)

    # Visualisasi distribusi data
    st.subheader("Class Distribution")
    fig = plt.figure(figsize=(30,12))
    plt.bar(np.unique(df["ClassName"], return_counts=True)[0], np.unique(df["ClassName"], return_counts=True)[1])
    plt.xticks(rotation=90, fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid()

    st.pyplot(fig)

    # Melihat distribusi size gambar
    st.subheader("Distribusi Sample Image Size")
    sizes=[]
    for img in train_files :
        temp_image = plt.imread(img)
        sizes.append((temp_image.shape[0], temp_image.shape[1]))
    widths, heights = zip(*sizes)
    fig2 = plt.figure(figsize=(30,12))
    plt.hist(widths, bins=30, alpha=0.5, label='Width')
    plt.hist(heights, bins=30, alpha=0.5, label='Height')
    plt.xlabel('Size')
    plt.ylabel('Frequency')
    plt.legend()
    st.pyplot(fig2)
