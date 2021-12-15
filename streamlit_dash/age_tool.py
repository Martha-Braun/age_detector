# import libraries
import os
import pandas as pd
import streamlit as st
import datetime
from PIL import Image
import cv2
import matplotlib as plt


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
                "Detect Faces",
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
            camera = cv2.VideoCapture(
                0
            )  # video capture source camera (Here webcam of laptop)

            while run:
                _, frame = camera.read()  # return a single frame in variable `frame`
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame)
                cv2.imwrite("streamlit_dash/pictures/picture_taken/c1.png", frame)
            else:
                st.write("Click Run to Start Webcam")

        # Option Predict Age
        if self.option == "Detect Faces":
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
                cv2.imwrite(
                    "pictures/faces_detected/face_" + str(idx) + "_" + image_file,
                    roi_color,
                )

            status = cv2.imwrite(
                "pictures/faces_detected/faces_detected.jpg", detected_image
            )

            st.image(detected_image)

        # Option Welcome
        if self.option == "Party Time!":
            st.markdown("## Party time!")
            st.write("#TGIF")
            btn = st.button("Celebrate!")
            if btn:
                st.balloons()
