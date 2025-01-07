import numpy as np
import cv2
import streamlit as st
from PIL import Image
import os

# Set page configuration to expand sidebar by default
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Function to create a sticky navigation bar
def render_navbar():
    navbar_style = """
    <style>
        .navbar {
            position: -webkit-sticky; /* Safari */
            position: sticky;
            top: 0;
            z-index: 1000;
            background-color: #0078D7;
            padding: 15px 10px;
            color: white;
            font-size: 18px;
            font-weight: bold;
            display: flex;
            justify-content: center;
        }
        .navbar a {
            color: white;
            text-decoration: none;
            margin: 0 20px;
            pointer-events: none; /* Disable clicking */
        }
    </style>
    <div class="navbar">
        <a href="#">Home</a>
        <a href="#">About</a>
        <a href="#">Contact</a>
    </div>
    """
    st.markdown(navbar_style, unsafe_allow_html=True)

# Function to add a copyright footer
def render_footer():
    footer_style = """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
            color: #333;
        }
    </style>
    <div class="footer">
        &copy; 2025 U Mahesh.
    </div>
    """
    st.markdown(footer_style, unsafe_allow_html=True)

# Function to center images
def center_image(image_html):
    return f"""
    <div style="display: flex; justify-content: center; margin: 20px 0;">
        {image_html}
    </div>
    """

# Render the navigation bar
render_navbar()

# Render the copyright footer
render_footer()

def colorizer(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    script_dir = os.path.dirname(__file__)
    prototxt = os.path.join(script_dir, "model/colorization_deploy_v2.prototxt")
    model = os.path.join(script_dir, "model/colorization_release_v2.caffemodel")
    points = os.path.join(script_dir, "model/pts_in_hull.npy")
    
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)
    
    # Add the cluster centers as 1x1 convolutions to the model
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    
    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
    
    L = cv2.split(lab)[0]
    st.markdown(center_image(st.image(L, caption="L Channel (Lightness)", width=600, clamp=True, channels="gray")), unsafe_allow_html=True)
    
    resized = cv2.resize(lab, (224, 224))
    L_resized = cv2.split(resized)[0]
    L_resized -= 50
    
    net.setInput(cv2.dnn.blobFromImage(L_resized))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
    
    st.markdown(center_image(st.image(ab[:, :, 0], caption="'a' Channel", width=600, clamp=True, channels="gray")), unsafe_allow_html=True)
    st.markdown(center_image(st.image(ab[:, :, 1], caption="'b' Channel", width=600, clamp=True, channels="gray")), unsafe_allow_html=True)
    
    ab_combined = np.zeros((ab.shape[0], ab.shape[1], 3))
    ab_combined[:, :, 0] = ab[:, :, 0]
    ab_combined[:, :, 1] = ab[:, :, 1]
    st.markdown(center_image(st.image(ab_combined, caption="Combined 'ab' Channels (Visualization)", width=600, clamp=True)), unsafe_allow_html=True)
    
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    
    return colorized

##########################################################################################################

# Streamlit App UI
st.title("Colorize Your Black and White Image")
st.write("This app colorizes your B&W images and displays intermediate phases like 'L', 'a', 'b', and 'ab' channels.")

# Upload image functionality
file = st.file_uploader("Upload an image file", type=["jpg", "png"])

if file is not None:
    image = Image.open(file)
    img = np.array(image)
    
    st.markdown(center_image(st.image(image, caption="Original Image", width=600)), unsafe_allow_html=True)
    
    st.text("Processing...")
    color = colorizer(img)
    
    st.markdown(center_image(st.image(color, caption="Colorized Image", width=600)), unsafe_allow_html=True)
else:
    st.text("Upload an image to display its colorized version.")

