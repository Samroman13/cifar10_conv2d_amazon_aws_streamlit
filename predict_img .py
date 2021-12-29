

from tensorflow.keras.models import load_model
MODEL_NAME =   'model_cf10.h5'
import numpy as np
from PIL import Image 
model = load_model(MODEL_NAME)                                              
INPUT_SHAPE = (32, 32, 3)
classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def process(image_file):
    
    image = Image.open(image_file)  # Opening the processed file
    resized_image = image.resize((INPUT_SHAPE[1], INPUT_SHAPE[0]))          # Resizing the image according to the network input
    array = np.array(resized_image)[..., :3][np.newaxis, ...]   # Adjusting the tensor shape for feeding to the network
    predict = model.predict(array)
    print(type(predict), predict)
    predict_text = classes[np.argmax(predict)]
  
    return resized_image, predict_text                          # Return of the original thumbnail image, text description of the prediction
