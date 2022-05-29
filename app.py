from typing import List, Optional
import streamlit as st
import streamlit_pydantic as sp
from pydantic import BaseModel, Field

from src.Surveyor import Surveyor



@st.experimental_singleton(show_spinner=True, suppress_st_warning=True)
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
    with open(str(zip_file_name), "rb") as file:
        btn = download_placeholder.download_button(
            label="Download extracted topic-clustered-highlights, images and tables as zip",
            data=file,
            file_name=str(zip_file_name)
        )

    with open(str(survey_file_name), "rb") as file:
        btn = download_placeholder.download_button(
            label="Download detailed generated survey file",
            data=file,
            file_name=str(survey_file_name)
        )


class KeywordsModel(BaseModel):
    research_keywords: Optional[str] =  Field(
        '', description="Enter your research keywords:"
    )
    max_search: int = Field(
        10, ge=1, le=50, multiple_of=1,
        description="num_papers_to_search:"
    )
    num_papers: int = Field(
        3, ge=1, le=8, multiple_of=1, 
        description="num_papers_to_select:"
    )


class ArxivIDsModel(BaseModel):
    arxiv_ids: Optional[str] =  Field(
        '', description="Enter comma_separated arxiv ids for your curated set of papers (e.g. 2205.12755, 2205.10937, ...):"
    )


def survey_space(surveyor, download_placeholder):
    with st.sidebar.form(key="survey_keywords_form"):
        session_data = sp.pydantic_input(key="keywords_input_model", model=KeywordsModel)
        st.write('or')
        session_data.update(sp.pydantic_input(key="arxiv_ids_input_model", model=ArxivIDsModel))
        submit = st.form_submit_button(label="Submit")
        
    run_kwargs = {'surveyor':surveyor, 'download_placeholder':download_placeholder}
    if submit:
        if session_data['research_keywords'] != '':
            run_kwargs.update({'research_keywords':session_data['research_keywords'], 
                               'max_search':session_data['research_keywords'], 
                               'num_papers':session_data['research_keywords']})
        elif session_data['arxiv_ids'] != '':
            run_kwargs.update({'arxiv_ids':[id.strip() for id in session_data['arxiv_ids'].split(',')]})
        st.json(run_kwargs)
        run_survey(**run_kwargs)


if __name__ == '__main__':
    st.title('Auto-Research V0.1 - Automated Survey generation from research keywords')
    std_col, survey_col = st.columns(2)
    std_col.header('execution log:')
    survey_col.header('Generated_survey:')
    download_placeholder = survey_col.container()
    surveyor_obj = get_surveyor_instance(_print_fn=std_col.write, _survey_print_fn=survey_col.write)
    survey_space(surveyor_obj, survey_col)
