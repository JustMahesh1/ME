import cv2
import numpy as np
import streamlit as st
from PIL import Image as PILImage
import tempfile
import os
from io import BytesIO
import glob
import zipfile

# Define paths to model files
prototxt = 'model/colorization_deploy_v2.prototxt'
model = 'model/colorization_release_v2.caffemodel'
points = 'model/pts_in_hull.npy'

# Load model files
net = cv2.dnn.readNetFromCaffe(prototxt, model)
pts = np.load(points)
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def colorize_image(image):
    st.subheader("Step 1: Original Image")
    st.image(image, use_column_width=True)

    # Convert image to float and LAB color space
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Normalize L, A, and B channels for display
    L_normalized = L / 255.0
    A_normalized = (A - A.min()) / (A.max() - A.min())  # Normalize A to [0, 1]
    B_normalized = (B - B.min()) / (B.max() - B.min())  # Normalize B to [0, 1]

    st.subheader("Step 2: Grayscale Image (L Channel)")
    st.image(L_normalized, channels="GRAY", use_column_width=True)

    st.subheader("Step 3: A and B Channels")
    st.image(A_normalized, channels="GRAY", use_column_width=True, caption="A Channel")
    st.image(B_normalized, channels="GRAY", use_column_width=True, caption="B Channel")

    # Resize L channel and adjust for model input
    resized_L = cv2.resize(L, (224, 224))
    L_resized = resized_L - 50
    L_resized = np.clip(L_resized, 0, 255)  # Ensure values are within [0, 255] range

    st.subheader("Step 4: Resized L Channel for Model Input")
    L_resized_display = L_resized / 255.0
    st.image(L_resized_display, channels="GRAY", use_column_width=True)

    # Set input for model and process
    net.setInput(cv2.dnn.blobFromImage(L_resized))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize predicted AB channels to original image size
    ab_resized = cv2.resize(ab, (image.shape[1], image.shape[0]))

    st.subheader("Step 5: Predicted AB Channels")
    ab_resized = (ab_resized - ab_resized.min()) / (ab_resized.max() - ab_resized.min())  # Normalize to [0, 1]
    ab_image = np.zeros_like(image)
    ab_image[:, :, 1:] = ab_resized
    ab_image = cv2.cvtColor(ab_image, cv2.COLOR_LAB2BGR)
    st.image(ab_image, channels="BGR", use_column_width=True)

    # Create final colorized image
    colorized = np.concatenate((L[:, :, np.newaxis], ab_resized), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)  # Ensure values are within [0, 1] for display
    colorized = (255 * colorized).astype("uint8")  # Convert to uint8 for display

    st.subheader("Step 6: Final Colorized Image")
    st.image(colorized, use_column_width=True)

    return colorized

def adjust_image(image, brightness=1.0, contrast=1.0, saturation=1.0, gamma=1.0):
    img = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img[..., 1] = hsv_img[..., 1] * saturation
    img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    img = np.power(img / 255.0, gamma)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img

def apply_filter(image, filter_type):
    if filter_type == 'Blur':
        return cv2.GaussianBlur(image, (15, 15), 0)
    elif filter_type == 'Sharpen':
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)
    elif filter_type == 'Edge Detection':
        return cv2.Canny(image, 100, 200)
    elif filter_type == 'Sepia':
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        return cv2.transform(image, sepia_filter)
    elif filter_type == 'Vignette':
        rows, cols = image.shape[:2]
        X_resultant_kernel = cv2.getGaussianKernel(cols, 200)
        Y_resultant_kernel = cv2.getGaussianKernel(rows, 200)
        resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T
        mask = 255 * resultant_kernel / np.linalg.norm(resultant_kernel)
        return cv2.filter2D(image, -1, mask)
    return image

def process_video(video_file, brightness, contrast, saturation, gamma, filter_type):
    video_file.seek(0)  # Reset file pointer to start
    temp_video_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name

    cap = cv2.VideoCapture(video_file.name)
    if not cap.isOpened():
        st.error("Error: Could not open the video file.")
        return None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        colorized_frame = colorize_image(frame)
        adjusted_frame = adjust_image(colorized_frame, brightness, contrast, saturation, gamma)
        final_frame = apply_filter(adjusted_frame, filter_type)
        out.write(final_frame)

    cap.release()
    out.release()
    return temp_video_path

def download_colorized_images(colorized_images, filenames):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for image, filename in zip(colorized_images, filenames):
            img_bytes = cv2.imencode('.png', image)[1].tobytes()
            zip_file.writestr(filename, img_bytes)
    zip_buffer.seek(0)
    return zip_buffer

st.title("High-End Photo & Video Colorizer")

uploaded_files = st.file_uploader("Upload images or video", type=["jpg", "jpeg", "png", "mp4", "avi"], accept_multiple_files=True)
folder_path = st.text_input("Enter folder path for images", "")

if uploaded_files or folder_path:
    images = []
    videos = []
    filenames = []

    # Process uploaded files
    for uploaded_file in uploaded_files:
        if uploaded_file.type.startswith('image'):
            image = np.array(PILImage.open(uploaded_file))
            images.append(image)
            filenames.append(uploaded_file.name)
        elif uploaded_file.type.startswith('video'):
            videos.append(uploaded_file)

    # Process images from a folder
    if folder_path:
        for file_path in glob.glob(os.path.join(folder_path, "*.jpg")):
            image = cv2.imread(file_path)
            images.append(image)
            filenames.append(os.path.basename(file_path))

    # Parameters
    st.sidebar.header("Adjustments")
    brightness = st.sidebar.slider("Brightness", -100.0, 100.0, 0.0, 0.1)
    contrast = st.sidebar.slider("Contrast", 0.0, 3.0, 1.0, 0.01)
    saturation = st.sidebar.slider("Saturation", 0.0, 3.0, 1.0, 0.01)
    gamma = st.sidebar.slider("Gamma", 0.1, 2.0, 1.0, 0.01)

    st.sidebar.header("Filters")
    filter_type = st.sidebar.selectbox("Filter", ["None", "Blur", "Sharpen", "Edge Detection", "Sepia", "Vignette"])

    colorized_images = []

    # Process each image
    for image in images:
        colorized = colorize_image(image)
        adjusted = adjust_image(colorized, brightness, contrast, saturation, gamma)
        final_image = apply_filter(adjusted, filter_type)
        st.image(final_image, use_column_width=True)
        colorized_images.append(final_image)

    # Download option for images
    if colorized_images:
        st.markdown("## Download Colorized Images")
        zip_buffer = download_colorized_images(colorized_images, filenames)
        st.download_button("Download All Images as ZIP", data=zip_buffer, file_name="colorized_images.zip")

    # Process each video
    for video in videos:
        processed_video_path = process_video(video, brightness, contrast, saturation, gamma, filter_type)
        if processed_video_path:
            st.markdown("## Download Processed Video")
            with open(processed_video_path, "rb") as file:
                st.download_button("Download Video", data=file, file_name=os.path.basename(processed_video_path))

    st.success("Processing complete!")

else:
    st.info("Please upload images or video files or specify a folder path to process.")
