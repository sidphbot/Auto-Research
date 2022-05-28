import streamlit as st
import pandas as pd
import numpy as np

from src.Surveyor import Surveyor

def run_survey(surveyor, research_keywords, max_search, num_papers):
    zip_file_name, survey_file_name = surveyor.survey(research_keywords, 
                                                  max_search=max_search, 
                                                  num_papers=num_papers
                                                )

    with open(str(zip_file_name), "rb") as file:
        btn = st.download_button(
             label="Download extracted topic-clustered-highlights, images and tables as zip",
             data=file,
             file_name=str(zip_file_name)
           )

    with open(str(survey_file_name), "rb") as file:
        btn = st.download_button(
             label="Download detailed generated survey file",
             data=file,
             file_name=str(survey_file_name)
           )
        for line in file.readlines():
            st.write(line)


def survey_space(surveyor):
    
    st.title('Automated Survey generation from research keywords - Auto-Research V0.1')

    form = st.sidebar.form(key='survey_form')
    research_keywords = form.text_input("What would you like to research in today?")
    max_search = form.number_input("num_papers_to_search", help="maximium number of papers to glance through - defaults to 20", 
                             min_value=1, max_value=60, value=20, step=1, key='max_search')
    num_papers = form.number_input("num_papers_to_select", help="maximium number of papers to select and analyse - defaults to 8",
                             min_value=1, max_value=25, value=8, step=1, key='num_papers')
    submit = form.form_submit_button('Submit')

    if submit:
        st.write("hello")
        run_survey(surveyor, research_keywords, max_search, num_papers)


if __name__ == '__main__':
    global surveyor
    surveyor_obj = Surveyor()
    survey_space(surveyor_obj)
