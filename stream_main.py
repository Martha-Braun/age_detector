from streamlit.age_tool import *
import streamlit as st


def main():
    dashboard = DashboardApp()

    # configure streamlit
    dashboard.configure_page()

    # create main pages
    dashboard.create_main_pages()


if __name__ == "__main__":
    main()
