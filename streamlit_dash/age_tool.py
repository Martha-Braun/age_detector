# import libraries
import os
import pandas as pd
import numpy as np
from skimage.color.colorconv import rgb2gray
import streamlit as st
import datetime
from PIL import Image
import cv2
import matplotlib as plt
from pathlib import Path

from skimage.color import gray2rgb
from skimage.color import rgb2gray
import tensorflow as tf
from keras.models import model_from_json
from skimage.transform import resize
from keras.applications.resnet import ResNet50
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.optimizers import Adam


class DashboardApp:
    def __init__(self):

        # to be created
        self.option = None

    def configure_page(self):
        """
        Configures app page
        Creates sidebar with selectbox leading to different main pages

        Returns:
            option (str): Name of main page selected by user
        """
        # set page configuration
        img_icon = Image.open(f"streamlit_dash/grandma.png")

        st.set_page_config(
            page_title="Age Detector",
            page_icon=img_icon,
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # create sidebar
        st.sidebar.title("Face Detector Dashboard")
        option = st.sidebar.selectbox(
            "Pick Dashboard:",
            (
                "Take Picture",
                "Detect Faces from Upload",
                "Detect Faces from WebCam",
                "Make Predictions",
                "Party Time!",
            ),
        )

        self.option = option

    def create_main_pages(self):
        """
        Creates pages for all options available in sidebar
        Page is loaded according to option picked in sidebar
        """

        # Option Main Page
        if self.option == "Take Picture":
            st.title("Step 1: Take a Picture")
            # st.markdown("Take a picture and we'll guess youre age")
            st.subheader("Smile!")
            run = st.checkbox("Run")
            FRAME_WINDOW = st.image([])
            camera = cv2.VideoCapture(0)  # video capture source (Here webcam of laptop)

            while run:
                _, frame = camera.read()  # return a single frame in variable `frame`
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame)
                color_conversion = cv2.COLOR_BGR2RGB  # color code to convert BGR to RGB
                im_rgb = cv2.cvtColor(frame, color_conversion)
                cv2.imwrite(
                    "streamlit_dash/pictures/picture_taken/c1.png", im_rgb
                )  # save image
            else:
                st.write("Click Run to Start Webcam")

        # Option Predict Age
        if self.option == "Detect Faces from Upload":
            st.title("Step 2: Detect Faces")

            uploaded_file = st.file_uploader(
                "Upload Image", type=["png", "jpg", "jpeg"]
            )

            if uploaded_file is not None:
                up_image = cv2.imread(
                    os.path.join(str(Path.home() / "Downloads"), uploaded_file.name)
                )
                gray = cv2.cvtColor(up_image, cv2.COLOR_BGR2GRAY)

                faceCascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                faces = faceCascade.detectMultiScale(
                    gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30)
                )

                detected_image = up_image.copy()
                for idx, (x, y, w, h) in enumerate(faces):
                    cv2.rectangle(
                        detected_image, (x, y), (x + w, y + h), (0, 0, 255), 2
                    )
                    roi_color = up_image[y : y + h, x : x + w]
                    print("[INFO] Object found. Saving locally.")
                    # save all detected faces individually
                    cv2.imwrite(
                        "streamlit_dash/pictures/faces_detected/face_"
                        + str(idx)
                        + "_"
                        + uploaded_file.name,
                        roi_color,
                    )

                # save image with faces marked
                status = cv2.imwrite(
                    "streamlit_dash/pictures/faces_marked/faces_detected.jpg",
                    detected_image,
                )

                # show uploaded image marking faces detected
                st.image(detected_image, channels="BGR")

        # Option Predict Age
        if self.option == "Detect Faces from WebCam":
            st.title("Step 2: Detect Faces")

            folder_path = "streamlit_dash/pictures/picture_taken"
            image_file = "c1.png"

            # take the image and convert it to an OpenCV object
            image = cv2.imread(os.path.join(folder_path, image_file))
            # convert it to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faceCascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = faceCascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30)
            )

            detected_image = image.copy()

            for idx, (x, y, w, h) in enumerate(faces):
                cv2.rectangle(detected_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                roi_color = image[y : y + h, x : x + w]
                print("[INFO] Object found. Saving locally.")
                # save all detected faces individually
                cv2.imwrite(
                    "streamlit_dash/pictures/faces_detected/face_"
                    + str(idx)
                    + "_"
                    + image_file,
                    roi_color,
                )

            # convert BGR to RGB code
            color_conversion = cv2.COLOR_BGR2RGB  # color code to convert BGR to RGB
            marked_img_rgb = cv2.cvtColor(detected_image, color_conversion)

            # save image with faces marked
            cv2.imwrite(
                "streamlit_dash/pictures/picture_marked/marked_img_rgb.jpg",
                marked_img_rgb,
            )

            # show uploaded image marking faces detected
            st.image(marked_img_rgb)

        # Option Make Predictions
        if self.option == "Make Predictions":

            # # load json and create model
            # json_file = open("streamlit_dash/model/model_mb.json", "r")
            # loaded_model_json = json_file.read()
            # json_file.close()
            # loaded_model = model_from_json(loaded_model_json)
            # # load weights into new model
            # loaded_model.load_weights("streamlit_dash/model/mb_model_weights.h5")
            # print("Loaded model from disk")

            model_dir = "streamlit_dash/model/"
            loaded_model = tf.keras.models.load_model(model_dir)

            img_detected_faces = os.listdir("streamlit_dash/pictures/faces_detected")

            for face in img_detected_faces:
                # get path of one image
                img_folder_path = "streamlit_dash/pictures/faces_detected/"
                img = cv2.imread(img_folder_path + face)

                # resize image to models input shape
                img_resized = resize(img, (48, 48))

                # transform to gray scale
                img_gray = rgb2gray(img_resized)

                # retransform to rgb to get right shape
                img_rgb = gray2rgb(img_gray)

                # add a fake dimension for a batch of 1
                img_batch = tf.expand_dims(img_rgb, axis=0)

                # make prediction
                predictions = loaded_model.predict(img_batch)
                st.image(img_folder_path + face, width=300)  # img_folder_path + face

                # st.markdown(f"Number of classes {predictions.shape[1]}")
                # dictionary classes to age
                age_buckets = {
                    0: "0-7",
                    1: "7-14",
                    2: "14-22",
                    3: "22-30",
                    4: "30-38",
                    5: "38-47",
                    6: "47-58",
                    7: "47-58",
                    8: "58-120",
                }

                st.markdown(
                    f"You're age range is {age_buckets[np.argmax(predictions)]}"
                )

        # Option Welcome
        if self.option == "Party Time!":
            st.markdown("## Party time!")
            st.write("#TGIF")
            btn = st.button("Celebrate!")
            if btn:
                st.balloons()
