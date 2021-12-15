# import libraries
import os
import pandas as pd
import streamlit as st
import datetime
from PIL import Image


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
        img_icon = Image.open(
            f"/Applications/documents bob/MSc Data Science/HEC/Deep Learning/lectures-labs/streamlit/grandma.png"
        )

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
            st.title("Age Detector")
            st.markdown("Take a picture and we'll guess youre age")

            st.subheader("Subheader")

        # Option Welcome
        if self.option == "Party Time!":
            st.markdown("## Party time!")
            st.write("#TGIF")
            btn = st.button("Celebrate!")
            if btn:
                st.balloons()
