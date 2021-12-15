from streamlit_dash.age_tool import *


def main():
    dashboard = DashboardApp()

    # configure streamlit
    dashboard.configure_page()

    # create main pages
    dashboard.create_main_pages()


if __name__ == "__main__":
    main()
