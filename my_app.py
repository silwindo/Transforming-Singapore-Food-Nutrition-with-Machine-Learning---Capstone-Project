
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
st.title("Image Classification")
upload_file = st.sidebar.file_uploader("Upload food images", type = 'jpg')
generate_pred = st.sidebar.button("predict")
model = tf.keras.models.load_model('vgg16emodel2.h5')
def import_n_pred(image_data, model):
    size = (224,224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    reshape = img[np.newaxis,...]
    pred = model.predict(reshape)
    return pred
if generate_pred:
    image = Image.open(upload_file)
    with st.expander('image', expanded=True):
        st.image(image, use_column_width=True)
    pred = import_n_pred(image, model)
    labels = ['ayam_goreng', 'ayam_pop', 'daging_rendang', 'dendeng_batokok', 'gulai_ikan', 'gulai_tambusu', 'gulai_tunjang', 'telur_balado', 'telur_dadar']
    st.title("prediction of image is {}".format(labels[np.argmax(pred)]))





