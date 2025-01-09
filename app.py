import numpy as np
import cv2
import streamlit as st
from PIL import Image
import os
import datetime

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


# Sidebar Functionality
def render_sidebar():
    st.sidebar.title("Menu")
    #st.sidebar.button("Home")
    #st.sidebar.button("About")
    #st.sidebar.button("Contact")


    # Home Button
    if st.sidebar.button("Home"):
        st.experimental_rerun()  # This will rerun the app

    # About Button with link redirection
    if st.sidebar.button("About"):
        st.write("[Go to About](https://instagram.com/just_mahesh_75)")  # Replace with actual link

    # Contact Button with link redirection
    if st.sidebar.button("Contact"):
        st.write("[Go to Contact Page](https://example.com/contact)")  # Replace with actual link

    help_expander = st.sidebar.expander("Help")
    with help_expander:
        st.write("""
            **How to use:**
            1. Upload a black and white image.
            2. The app will colorize the image and display the result.
            3. You can see intermediate steps like the 'L', 'a', 'b' channels.
            4. History shows all previously uploaded images.
        """)


    # History Expander
    history_expander = st.sidebar.expander("History")
    with history_expander:
        if 'history' not in st.session_state:
            st.session_state.history = []
        
        # Display history if available
        if st.session_state.history:
            for idx, (image, timestamp) in enumerate(st.session_state.history):
                st.image(image, use_column_width=True, caption=f"Image {idx + 1} - Uploaded at {timestamp}")
        else:
            st.write("No history available. Upload an image to start.")

# Function to handle multiple image uploads and colorization
def process_images(files):
    history_images = []
    for file in files:
        img = Image.open(file)
        img = np.array(img)

        # Process and colorize the image
        color = colorizer(img)

        # Add to history with timestamp (only the uploaded images, not colorized)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.history.append((file, timestamp))
        history_images.append(file)
        
        # Display the image and the colorized version
        st.text(f"Uploaded Image {len(st.session_state.history)}")
        st.image(img, width=600)
        st.text(f"Colorized Image {len(st.session_state.history)}")
        st.image(color, width=600)

    return history_images

# Function to colorize the image
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

# Streamlit App UI
st.title("Colorize Your Black and White Image")
st.write("This app colorizes your B&W images and displays intermediate phases like 'L', 'a', 'b' and 'ab' channels.")

# Multiple image upload functionality
files = st.file_uploader("Upload image files", type=["jpg", "png"], accept_multiple_files=True)

if files:
    # Process and display uploaded files
    history_images = process_images(files)
else:
    st.text("Upload images to display their colorized versions.")

# Render the sidebar navigation
render_sidebar()

