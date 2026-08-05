import streamlit as st


pages = [
    st.Page("Home.py", title="Home", default=True),
    st.Page("pages/1_Draft_Mode.py", title="Draft Mode"),
    st.Page("pages/2_Player_Cards.py", title="Player Cards"),
    st.Page("pages/3_Team_Outlook.py", title="Team Outlook"),
    st.Page("pages/5_Player_Compare.py", title="Player Compare"),
    st.Page("pages/7_Live_Rankings.py", title="Live Rankings"),
    st.Page("pages/8_Tier_Desperation.py", title="Tier Desperation"),
    st.Page("pages/9_Component_Audit.py", title="Component Audit"),
]

page = st.navigation(pages)
page.run()
