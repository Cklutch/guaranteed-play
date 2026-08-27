import streamlit as st


pages = [
    st.Page("Home.py", title="Home", default=True),
    st.Page("pages/1_Draft_Mode.py", title="Draft Mode"),
    st.Page("pages/3_Team_Outlook.py", title="Team Outlook"),
]

page = st.navigation(pages)
page.run()
