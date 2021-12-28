

import streamlit as st
from PIL import Image 
from predict_img import process

st.title('Cifar_10 demo')

image_file = st.file_uploader('Load an image', type=['png', 'jpg'])  # Adding a file loader

if not image_file is None:                                           # Block execution if an image is loaded
    col1, col2 = st.beta_columns(2)                                  # Creating 2 columns
    image = Image.open(image_file)                                   # Image opening
    pred_img, pred_text,  = process(image_file)                      # Image processing using a function implemented in another file
    col1.text('Source image')
    col1.image(image_file)                                           # Output in the first column of the original image
    col2.text(pred_text)                                             # Output of the text prediction of the model
    col2.image(pred_img)                                             # Output of the reduced source image to the second column
    
