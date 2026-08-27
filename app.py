import streamlit as st


pages = [
    st.Page("Home.py", title="Home", default=True),
    st.Page("pages/1_Draft_Mode.py", title="Draft Mode"),
    st.Page("pages/3_Team_Outlook.py", title="Team Outlook"),
    st.Page("pages/9_Component_Audit.py", title="Component Audit"),
]

page = st.navigation(pages)
page.run()
