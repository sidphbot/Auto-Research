import streamlit as st
import pandas as pd
import numpy as np

from src.Surveyor import Surveyor

import contextlib
from functools import wraps
from io import StringIO

def capture_output(func):
    """Capture output from running a function and write using streamlit."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Redirect output to string buffers
        stdout, stderr = StringIO(), StringIO()
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                return func(*args, **kwargs)
        except Exception as err:
            st.write(f"Failure while executing: {err}")
        finally:
            if _stdout := stdout.getvalue():
                st.write("Execution stdout:")
                st.code(_stdout)
            if _stderr := stderr.getvalue():
                st.write("Execution stderr:")
                st.code(_stderr)

    return wrapper

def run_survey(surveyor, research_keywords, max_search, num_papers):
    survey_fn = capture_output(surveyor.survey)
    zip_file_name, survey_file_name = survey_fn(research_keywords, 
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
    st.sidebar.title('Auto-Research V0.1 - Automated Survey generation from research keywords')
    form = st.sidebar.form(key='survey_form')
    research_keywords = form.text_input("What would you like to research in today?")
    max_search = form.number_input("num_papers_to_search", help="maximium number of papers to glance through - defaults to 20", 
                             min_value=1, max_value=60, value=10, step=1, key='max_search')
    num_papers = form.number_input("num_papers_to_select", help="maximium number of papers to select and analyse - defaults to 8",
                             min_value=1, max_value=25, value=2, step=1, key='num_papers')
    submit = form.form_submit_button('Submit')

    if submit:
        st.write("hello")
        run_survey(surveyor, research_keywords, max_search, num_papers)


if __name__ == '__main__':
    global surveyor
    surveyor_obj = Surveyor()
    survey_space(surveyor_obj)
