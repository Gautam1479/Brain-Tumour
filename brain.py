import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Ensure set_page_config is the first Streamlit command
st.set_page_config("Brain Tumor Detection", page_icon=":brain:")
st.title("Brain Tumor Detector using CNN")

st.write("""
Brain tumors are one of the most critical health issues, requiring early and accurate detection for effective treatment.
Traditional methods of brain tumor detection involve manual inspection of MRI images by radiologists,
which can be time-consuming and prone to human error. With the advent of deep learning techniques, 
automated brain tumor detection has become a promising approach.
""")

st.write("""
Deep learning models, particularly convolutional neural networks (CNNs), have shown remarkable success in medical image analysis.
These models can learn complex patterns in MRI images and accurately classify regions as tumor or non-tumor. The process typically 
involves pre-processing MRI images, training a deep neural network using libraries like TensorFlow and Keras, and evaluating 
the model's performance using metrics such as accuracy, sensitivity, and specificity.
""")

st.subheader("Types of Brain Tumors")
st.write("""
Brain tumors can be broadly categorized into primary and secondary tumors. Primary brain tumors originate within the brain, 
whereas secondary (metastatic) tumors spread to the brain from other parts of the body.
""")

st.write("""
**Gliomas**: These are the most common type of primary brain tumors, arising from glial cells. 
They include subtypes like astrocytomas, oligodendrogliomas, and ependymomas.
**Meningiomas**: These tumors develop from the meninges, the protective layers surrounding the brain and spinal cord. 
They are usually benign but can occasionally be malignant.
**Pituitary Tumors**: Originating in the pituitary gland, these tumors can affect hormone production and cause various systemic symptoms.
**Schwannomas**: These tumors develop from Schwann cells, which produce the myelin sheath covering nerves. An example is the acoustic neuroma, which affects hearing and balance.
**Medulloblastomas**: These are highly malignant tumors that are most commonly found in children and originate in the cerebellum.
""")

st.write("Please fill in the details below for the brain tumor detection analysis.")

patient_name = st.text_input("Patient Name")
age = st.number_input("Age", min_value=0, max_value=120, step=1)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
mri_image = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg", "dcm"])

if st.button('Submit'):
    if patient_name and age and gender and mri_image is not None:
        st.success(f"Form submitted successfully for {patient_name}")
        # Load image
        img = image.load_img(mri_image, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        model = load_model('brainT (1).h5')
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        
        f = predicted_class[0]
        print(f"Predicted class: {f}")
        
        if f == 2:
            message = f"{patient_name}, you are safe. You do not have a tumor."
            st.success(message)
            print(message)
        else:
            if f == 0:
                a = "Glioma Tumor"
            elif f == 1:
                a = "Meningioma Tumor"
            elif f == 3:
                a = "Pituitary Tumor"
            
            message = f"Sad to say, {patient_name}, but it seems like you are suffering from a {a}."
            st.error(message)
            print(message)
    else:
        st.error("Please fill in all the fields before submitting")
