# # Import the necessary packages
# import numpy as np
# import cv2
# import streamlit as st
# from PIL import Image
# import os
# from streamlit import *
# import base64

# # Set page configuration to expand sidebar by default
# st.set_page_config(layout="wide", initial_sidebar_state="expanded")


# # Function to create a navigation bar with padding
# def render_navbar():
#     navbar_style = """
#     <style>
#         .navbar {
#             background-color: #0078D7;
#             padding: 15px 10px;
#             color: white;
#             font-size: 18px;
#             font-weight: bold;
#         }
#         .navbar a {
#             color: white;
#             text-decoration: none;
#             margin-right: 20px;
#         }
#         .navbar a:hover {
#             text-decoration: underline;
#         }
#     </style>
#     <div class="navbar">
#         <a href="#Colorize Your Black and White Image">Home</a>
#         <a href="#About">About</a>
#         <a href="#Contact">Contact</a>
#     </div>
#     """
#     st.markdown(navbar_style, unsafe_allow_html=True)



# def set_background_image(image_path):
#     # Open the image file and encode it into base64
#     with open(image_path, "rb") as image_file:
#         encoded_image = base64.b64encode(image_file.read()).decode()
    
#     # Construct the background style using the base64-encoded image
#     background_style = f"""
#     <style>
#         .stApp {{
#             background-image: url('data:image/jpg;base64,{encoded_image}');
#             background-size: cover;
#             background-position: center;
#             background-attachment: fixed;
#         }}
#     </style>
#     """
#     st.markdown(background_style, unsafe_allow_html=True)

# # Set background image (local path)
# set_background_image("D:/FY_TEST/templates/color-back.jpg")  # Make sure the path is correct


# # Render the navigation bar
# render_navbar()


# def colorizer(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
#     # Use relative paths based on the current script directory
#     script_dir = os.path.dirname(__file__)
#     prototxt = os.path.join(script_dir, "model/colorization_deploy_v2.prototxt")
#     model = os.path.join(script_dir, "model/colorization_release_v2.caffemodel")
#     points = os.path.join(script_dir, "model/pts_in_hull.npy")
    
#     net = cv2.dnn.readNetFromCaffe(prototxt, model)
#     pts = np.load(points)
    
#     # Add the cluster centers as 1x1 convolutions to the model
#     class8 = net.getLayerId("class8_ab")
#     conv8 = net.getLayerId("conv8_313_rh")
#     pts = pts.transpose().reshape(2, 313, 1, 1)
#     net.getLayer(class8).blobs = [pts.astype("float32")]
#     net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    
#     # Scale the pixel intensities to the range [0, 1], and convert the image from BGR to Lab color space
#     scaled = img.astype("float32") / 255.0
#     lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
    
#     # Display the L channel
#     L = cv2.split(lab)[0]
#     st.image(L, caption="L Channel (Lightness)", width=600, clamp=True, channels="gray")
    
#     # Resize the Lab image to 224x224 (the dimensions the colorization network accepts), extract the 'L' channel, and perform mean centering
#     resized = cv2.resize(lab, (224, 224))
#     L_resized = cv2.split(resized)[0]
#     L_resized -= 50
    
#     # Pass the L channel through the network which will predict the 'a' and 'b' channel values
#     net.setInput(cv2.dnn.blobFromImage(L_resized))
#     ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    
#     # Resize the predicted 'ab' volume to the same dimensions as the input image
#     ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
    
#     # Display the 'a' and 'b' channels separately as grayscale images
#     st.image(ab[:, :, 0], caption="'a' Channel", width=600, clamp=True, channels="gray")
#     st.image(ab[:, :, 1], caption="'b' Channel", width=600, clamp=True, channels="gray")
    
#     # Combine 'ab' channels into a dummy 3-channel image for visualization
#     ab_combined = np.zeros((ab.shape[0], ab.shape[1], 3))
#     ab_combined[:, :, 0] = ab[:, :, 0]  # Map 'a' to the red channel
#     ab_combined[:, :, 1] = ab[:, :, 1]  # Map 'b' to the green channel
#     st.image(ab_combined, caption="Combined 'ab' Channels (Visualization)", width=600, clamp=True)
    
#     # Grab the 'L' channel from the original input image and concatenate it with the predicted 'ab' channels
#     L = cv2.split(lab)[0]
#     colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    
#     # Convert the output image from Lab color space to RGB, and clip values to [0, 1]
#     colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
#     colorized = np.clip(colorized, 0, 1)
    
#     # Convert to unsigned 8-bit integer representation in the range [0, 255]
#     colorized = (255 * colorized).astype("uint8")
    
#     return colorized

# ##########################################################################################################

# # Streamlit App UI
# st.title("Colorize Your Black and White Image")
# st.write("This app colorizes your B&W images and displays intermediate phases like 'L', 'a', 'b', and 'ab' channels.")

# # Load sample images
# input_images_dir = "C:/Users/mahes/Desktop/New folder/"
# input_images = [f for f in os.listdir(input_images_dir) if f.endswith(('.jpg', '.png'))]
# selected_image = st.sidebar.selectbox("Choose a sample image", ["None"] + input_images)

# file = st.sidebar.file_uploader("Or upload an image file", type=["jpg", "png"])

# if file is None:
#     if selected_image != "None":
#         image_path = os.path.join(input_images_dir, selected_image)
#         image = Image.open(image_path)
#         img = np.array(image)
#     else:
#         image = None
# else:
#     image = Image.open(file)
#     img = np.array(image)

# if image:
#     st.text("Your Original Image")
#     st.image(image, width=600)
    
#     st.text("Processing...")
#     color = colorizer(img)
    
#     st.text("Your Colorized Image")
#     st.image(color, width=600)
# else:
#     st.text("Select an image to display its colorized version.")


import numpy as np
import cv2
import streamlit as st
from PIL import Image
import os

# Set page configuration with favicon and layout settings
st.set_page_config(
    page_title="Colorizer App",
    page_icon="color-revive.ico",  # Add a favicon or path to an image file
    layout="wide",
    initial_sidebar_state="expanded"
)



# Add background gradient and favicon styling
def apply_custom_styling():
    gradient_style = """
    <style>
        body {
            background: linear-gradient(135deg, #1e3c72, #2a5298, #1e3c72);
            color: white;
            font-family: Arial, sans-serif;
        }

        .css-1d391kg { /* Sidebar adjustments */
            background-color: #2C3E50;
            color: white;
        }

        footer {
            color: white;
            background-color: #2C3E50;
            border-top: 1px solid #ECF0F1;
            padding: 10px;
        }

        header {
            color: white;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }

        .stTextInput input {
            background-color: #34495E;
            color: white;
        }

        .stButton button {
            background-color: #16A085;
            color: white;
            font-weight: bold;
            border-radius: 5px;
        }

        .stButton button:hover {
            background-color: #1ABC9C;
            transform: scale(1.05);
            transition: all 0.3s ease;
        }
    </style>
    """
    st.markdown(gradient_style, unsafe_allow_html=True)

apply_custom_styling()


# Function to create a beautiful sidebar
def render_sidebar():
    sidebar_style = """
    <style>
        /* Sidebar Style */
        .css-1d391kg {
            background-color: #2C3E50;
            color: white;
            padding: 20px;
            border-radius: 15px;
        }

        .sidebar .sidebar-content {
            padding: 0;
        }

        .sidebar .sidebar-content a {
            display: block;
            font-size: 18px;
            font-weight: bold;
            color: #ECF0F1;
            text-decoration: none;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            transition: all 0.3s ease;
        }

        .sidebar .sidebar-content a:hover {
            background-color: #16A085;
            color: white;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
        }

        .sidebar .sidebar-content a.active {
            background-color: #1ABC9C;
            color: white;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
        }
    </style>
    """
    st.markdown(sidebar_style, unsafe_allow_html=True)

# Function to create a sticky sidebar with a collapsible menu
def render_sidebar():
    # Sidebar content
    st.sidebar.title("Menu")
    st.sidebar.button("Home")
    st.sidebar.button("About")
    st.sidebar.button("Contact")

# Function to add a fixed copyright footer
def render_footer():
    footer_style = """
    <style>
        footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
            z-index: 1000;
            box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.1);
        }
    </style>
    <footer>
        &copy; 2025 U Mahesh.
    </footer>
    """
    st.markdown(footer_style, unsafe_allow_html=True)

# Render the sidebar navigation
render_sidebar()

# Render the copyright footer
render_footer()

def colorizer(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Use relative paths based on the current script directory
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
    
    # Scale the pixel intensities to the range [0, 1], and convert the image from BGR to Lab color space
    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
    
    # Display the L channel
    L = cv2.split(lab)[0]
    st.image(L, caption="L Channel (Lightness)", width=600, clamp=True, channels="gray")
    
    # Resize the Lab image to 224x224 (the dimensions the colorization network accepts), extract the 'L' channel, and perform mean centering
    resized = cv2.resize(lab, (224, 224))
    L_resized = cv2.split(resized)[0]
    L_resized -= 50
    
    # Pass the L channel through the network which will predict the 'a' and 'b' channel values
    net.setInput(cv2.dnn.blobFromImage(L_resized))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    
    # Resize the predicted 'ab' volume to the same dimensions as the input image
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
    
    # Display the 'a' and 'b' channels separately as grayscale images
    st.image(ab[:, :, 0], caption="'a' Channel", width=600, clamp=True, channels="gray")
    st.image(ab[:, :, 1], caption="'b' Channel", width=600, clamp=True, channels="gray")
    
    # Combine 'ab' channels into a dummy 3-channel image for visualization
    ab_combined = np.zeros((ab.shape[0], ab.shape[1], 3))
    ab_combined[:, :, 0] = ab[:, :, 0]  # Map 'a' to the red channel
    ab_combined[:, :, 1] = ab[:, :, 1]  # Map 'b' to the green channel
    st.image(ab_combined, caption="Combined 'ab' Channels (Visualization)", width=600, clamp=True)
    
    # Grab the 'L' channel from the original input image and concatenate it with the predicted 'ab' channels
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    
    # Convert the output image from Lab color space to RGB, and clip values to [0, 1]
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    
    # Convert to unsigned 8-bit integer representation in the range [0, 255]
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
    
    st.text("Your Original Image")
    st.image(image, width=600)
    
    st.text("Processing...")
    color = colorizer(img)
    
    st.text("Your Colorized Image")
    st.image(color, width=600)
else:
    st.text("Upload an image to display its colorized version.")
