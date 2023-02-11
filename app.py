import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# Define the URL of the prediction service
url = "https://0ae0s4rn1g.execute-api.us-west-2.amazonaws.com/test"

# Define the input form for the Streamlit app
st.header("Image Classifier")
image_url = st.text_input("Enter the URL of an image to classify")

# Make a prediction when the user clicks the "Predict" button
if st.button("Predict"):
    # Send a request to the prediction service with the input image URL
    data = {"url": image_url}
    response = requests.post(url, json=data).json()
    # Display the predicted class and probability
    st.write(f"Predicted class: {response}")
    # Load the image from the URL and display it
    image = Image.open(BytesIO(requests.get(image_url).content))
    st.image(image, caption="Input image")
    
