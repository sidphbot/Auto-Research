import streamlit as st
import pandas as pd
import numpy as np

from src.Surveyor import Surveyor
from streamlit_tags import st_tags_sidebar


@st.experimental_singleton
def get_surveyor_instance(_print_fn, _survey_print_fn):
     with st.spinner('Loading The-Surveyor ...'):
        return Surveyor(print_fn=_print_fn, survey_print_fn=_survey_print_fn, high_gpu=True)


def run_survey(surveyor, download_placeholder, research_keywords=None, arxiv_ids=None, max_search=None, num_papers=None):
    zip_file_name, survey_file_name = surveyor.survey(research_keywords, 
                                                      arxiv_ids,
                                                      max_search=max_search, 
                                                      num_papers=num_papers
                                                     )
    show_survey_download(zip_file_name, survey_file_name, download_placeholder)


def show_survey_download(zip_file_name, survey_file_name, download_placeholder):
    download_placeholder.empty()
    with download_placeholder.container():
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


def survey_space(surveyor, download_placeholder):

    form = st.sidebar.form(key='survey_form')
    research_keywords = form.text_input("What would you like to research in today?", key='research_keywords')
    max_search = form.number_input("num_papers_to_search", help="maximium number of papers to glance through - defaults to 20", 
                             min_value=1, max_value=50, value=10, step=1, key='max_search')
    num_papers = form.number_input("num_papers_to_select", help="maximium number of papers to select and analyse - defaults to 8",
                             min_value=1, max_value=8, value=2, step=1, key='num_papers')
    submit = form.form_submit_button('Submit')

    st.sidebar.write('or')

    arxiv_ids = st_tags_sidebar(
                label='# Enter Keywords:',
                value=[],
                text='Press enter to add more',
                maxtags = 6,
                key='arxiv_ids')

    if submit:
        run_survey(surveyor, download_placeholder, research_keywords, max_search, num_papers)
    elif len(arxiv_ids):
        run_survey(surveyor, download_placeholder, arxiv_ids)




if __name__ == '__main__':
    st.title('Auto-Research V0.1 - Automated Survey generation from research keywords')
    std_col, survey_col = st.columns(2)
    std_col.header('execution log:')
    survey_col.header('Generated_survey:')
    download_placeholder = survey_col.container()
    download_placeholder = st.empty()
    surveyor_obj = get_surveyor_instance(_print_fn=std_col.write, _survey_print_fn=survey_col.write)
    survey_space(surveyor_obj, survey_col)
