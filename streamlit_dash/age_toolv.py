# import libraries
import os
import pandas as pd
import numpy as np
import streamlit as st
import datetime
from PIL import Image
import cv2
from pathlib import Path
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
                "Detect Faces from Camera",
                "Detect Faces from Upload",
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
            camera = cv2.VideoCapture(
                0
            )  # video capture source camera (Here webcam of laptop)
            while run:
                _, frame = camera.read()  # return a single frame in variable `frame`
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame)
                
                cv2.imwrite("streamlit_dash/pictures/picture_taken/c1.png", frame)
            else:
                st.write("Click Run to Start Webcam and Stop to take the picture")

        if self.option == "Detect Faces from Upload":
            st.title("Step 2: Detect Faces")
            
            uploaded_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
            
            if uploaded_file is not None:                  
                up_image = cv2.imread(os.path.join(str(Path.home() / "Downloads"), uploaded_file.name))
                # st.image(up_image, channels="BGR", width=400) # view image
                
                gray = cv2.cvtColor(up_image, cv2.COLOR_BGR2GRAY)

                faceCascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                faces = faceCascade.detectMultiScale(
                    gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30)
                )

                detected_image = up_image.copy()
                for idx, (x, y, w, h) in enumerate(faces):
                    cv2.rectangle(detected_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
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
        if self.option == "Detect Faces from Camera":
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

            # save image with faces marked
            status = cv2.imwrite(
                "streamlit_dash/pictures/faces_marked/faces_detected.jpg",
                detected_image,
            )

            # show uploaded image marking faces detected
            st.image(detected_image) 
                    

        # Option Welcome
        if self.option == "Party Time!":
            st.markdown("## Party time!")
            st.write("#TGIF")
            btn = st.button("Celebrate!")
            if btn:
                st.balloons()
