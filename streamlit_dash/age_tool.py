# import libraries
import os
import pandas as pd
import streamlit as st
import datetime
from PIL import Image
import cv2


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
                "Main Page",
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
        if self.option == "Main Page":
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
                cv2.imwrite("streamlit_dash/c1.png", frame)
            else:
                st.write("Stopped")

            st.title("Step 2: Predict Age")

        # Option Welcome
        if self.option == "Party Time!":
            st.markdown("## Party time!")
            st.write("#TGIF")
            btn = st.button("Celebrate!")
            if btn:
                st.balloons()
