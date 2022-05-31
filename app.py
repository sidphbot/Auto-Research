from typing import List, Optional
import streamlit as st
import streamlit_pydantic as sp
from pydantic import BaseModel, Field
from PIL import Image
import tempfile
from pathlib import Path

from src.Surveyor import Surveyor


@st.experimental_singleton(suppress_st_warning=True)
def get_surveyor_instance(_print_fn, _survey_print_fn):
     with st.spinner('Loading The-Researcher ...'):
        return Surveyor(print_fn=_print_fn, survey_print_fn=_survey_print_fn, high_gpu=True)


def run_survey(surveyor, download_placeholder, research_keywords=None, arxiv_ids=None, max_search=None, num_papers=None):
    import hashlib
    import time

    hash = hashlib.sha1()
    hash.update(str(time.time()))
    temp_hash = hash.hexdigest()
    survey_root = Path(temp_hash).resolve()
    dir_args = {f'{dname}_dir': survey_root / dname for dname in ['pdf', 'txt', 'img', 'tab', 'dump']}
    for d in dir_args.values():
        d.mkdir(exist_ok=True, parents=True)
    print(survey_root)
    print(dir_args)
    dir_args = {k: str(v.resolve()) for k, v in dir_args.items()}
    zip_file_name, survey_file_name = surveyor.survey(research_keywords, 
                                                        arxiv_ids,
                                                        max_search=max_search, 
                                                        num_papers=num_papers
                                                        **dir_args)
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

if __name__ == '__main__':
    st.sidebar.image(Image.open('logo_landscape.png'), use_column_width = 'always')
    st.title('Auto-Research')
    st.write('#### A no-code utility to generate a detailed well-cited survey with topic clustered sections' 
             '(draft paper format) and other interesting artifacts from a single research query or a curated set of papers(arxiv ids).')
    st.write('##### Data Provider: arXiv Open Archive Initiative OAI')
    st.write('##### GitHub: https://github.com/sidphbot/Auto-Research')
    download_placeholder = st.container()

    with st.sidebar.form(key="survey_keywords_form"):
        session_data = sp.pydantic_input(key="keywords_input_model", model=KeywordsModel)
        st.write('or')
        session_data.update(sp.pydantic_input(key="arxiv_ids_input_model", model=ArxivIDsModel))
        submit = st.form_submit_button(label="Submit")
    st.sidebar.write('#### execution log:')
        
    run_kwargs = {'surveyor':get_surveyor_instance(_print_fn=st.sidebar.write, _survey_print_fn=st.write),
                  'download_placeholder':download_placeholder}
    if submit:
        if session_data['research_keywords'] != '':
            run_kwargs.update({'research_keywords':session_data['research_keywords'], 
                               'max_search':session_data['max_search'], 
                               'num_papers':session_data['num_papers']})
        elif session_data['arxiv_ids'] != '':
            run_kwargs.update({'arxiv_ids':[id.strip() for id in session_data['arxiv_ids'].split(',')]})

        #run_survey(**run_kwargs)
        
