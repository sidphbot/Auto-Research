# Auto-Research
A no-code utility to generate a detailed well-cited survey with topic clustered sections (draft paper format) and other interesting artifacts from a single research query.

Data Provider: [arXiv](https://arxiv.org/) Open Archive Initiative OAI

Requires:
 - python 3.7 or above
 - poppler-utils
 - list of requirements in requirements.txt
 - 8GB disk space 
 - 13GB CUDA(GPU) memory - for a survey of 100 searched papers(max_search) and 25 selected papers(num_papers)

#### Demo : 

Video Demo : (https://drive.google.com/file/d/1-77J2L10lsW-bFDOGdTaPzSr_utY743g/view?usp=sharing)

Kaggle descriptive re-usable Demo : https://www.kaggle.com/sidharthpal/auto-research-generate-survey-from-query 

(`[TIP]` click 'edit and run' to run the demo for your custom queries on a free GPU)

#### Steps to run (pip coming soon):
```
apt install -y poppler-utils libpoppler-cpp-dev
git clone https://github.com/sidphbot/Auto-Research.git

cd Auto-Research/
pip install -r requirements.txt
python survey.py [options] <your_research_query>
```

#### Artifacts generated (zipped):
- Detailed survey draft paper as txt file
- A curated list of top 25+ papers as pdfs and txts
- Images extracted from above papers as jpegs, bmps etc
- Heading/Section wise highlights extracted from above papers as a re-usable pure python joblib dump
- Tables extracted from papers(optional)
- Corpus of metadata highlights/text of top 100 papers as a re-usable pure python joblib dump

## Example run #1 - python utility

```
python survey.py 'multi-task representation learning'
```

## Example run #2 - python class

```
from survey import Surveyor
mysurveyor = Surveyor()
mysurveyor.survey('quantum entanglement')
```

## Access/Modify defaults:

- inside code 
```
from survey.Surveyor import DEFAULTS
from pprint import pprint

pprint(DEFAULTS)
```
or,

- Modify static config file - `defaults.py`

or,

- At runtime (utility)

```
python survey.py --help
```
```
usage: survey.py [-h] [--max_search max_metadata_papers]
                   [--num_papers max_num_papers] [--pdf_dir pdf_dir]
                   [--txt_dir txt_dir] [--img_dir img_dir] [--tab_dir tab_dir]
                   [--dump_dir dump_dir] [--models_dir save_models_dir]
                   [--title_model_name title_model_name]
                   [--ex_summ_model_name extractive_summ_model_name]
                   [--ledmodel_name ledmodel_name]
                   [--embedder_name sentence_embedder_name]
                   [--nlp_name spacy_model_name]
                   [--similarity_nlp_name similarity_nlp_name]
                   [--kw_model_name kw_model_name]
                   [--refresh_models refresh_models] [--high_gpu high_gpu]
                   query_string

Generate a survey just from a query !!

positional arguments:
  query_string          your research query/keywords

optional arguments:
  -h, --help            show this help message and exit
  --max_search max_metadata_papers
                        maximium number of papers to gaze at - defaults to 100
  --num_papers max_num_papers
                        maximium number of papers to download and analyse -
                        defaults to 25
  --pdf_dir pdf_dir     pdf paper storage directory - defaults to
                        arxiv_data/tarpdfs/
  --txt_dir txt_dir     text-converted paper storage directory - defaults to
                        arxiv_data/fulltext/
  --img_dir img_dir     image storage directory - defaults to
                        arxiv_data/images/
  --tab_dir tab_dir     tables storage directory - defaults to
                        arxiv_data/tables/
  --dump_dir dump_dir   all_output_dir - defaults to arxiv_dumps/
  --models_dir save_models_dir
                        directory to save models (> 5GB) - defaults to
                        saved_models/
  --title_model_name title_model_name
                        title model name/tag in hugging-face, defaults to
                        'Callidior/bert2bert-base-arxiv-titlegen'
  --ex_summ_model_name extractive_summ_model_name
                        extractive summary model name/tag in hugging-face,
                        defaults to 'allenai/scibert_scivocab_uncased'
  --ledmodel_name ledmodel_name
                        led model(for abstractive summary) name/tag in
                        hugging-face, defaults to 'allenai/led-
                        large-16384-arxiv'
  --embedder_name sentence_embedder_name
                        sentence embedder name/tag in hugging-face, defaults
                        to 'paraphrase-MiniLM-L6-v2'
  --nlp_name spacy_model_name
                        spacy model name/tag in hugging-face (if changed -
                        needs to be spacy-installed prior), defaults to
                        'en_core_sci_scibert'
  --similarity_nlp_name similarity_nlp_name
                        spacy downstream model(for similarity) name/tag in
                        hugging-face (if changed - needs to be spacy-installed
                        prior), defaults to 'en_core_sci_lg'
  --kw_model_name kw_model_name
                        keyword extraction model name/tag in hugging-face,
                        defaults to 'distilbert-base-nli-mean-tokens'
  --refresh_models refresh_models
                        Refresh model downloads with given names (needs
                        atleast one model name param above), defaults to False
  --high_gpu high_gpu   High GPU usage permitted, defaults to False

```

- At runtime (code)

    > during surveyor object initialization with `surveyor_obj = Surveyor()`
    - `pdf_dir`: String, pdf paper storage directory - defaults to `arxiv_data/tarpdfs/`
    - `txt_dir`: String, text-converted paper storage directory - defaults to `arxiv_data/fulltext/`
    - `img_dir`: String, image image storage directory - defaults to `arxiv_data/images/`
    - `tab_dir`: String, tables storage directory - defaults to `arxiv_data/tables/`
    - `dump_dir`: String, all_output_dir - defaults to `arxiv_dumps/`
    - `models_dir`: String, directory to save to huge models, defaults to `saved_models/`
    - `title_model_name`: String, title model name/tag in hugging-face, defaults to `Callidior/bert2bert-base-arxiv-titlegen`
    - `ex_summ_model_name`: String, extractive summary model name/tag in hugging-face, defaults to `allenai/scibert_scivocab_uncased`
    - `ledmodel_name`: String, led model(for abstractive summary) name/tag in hugging-face, defaults to `allenai/led-large-16384-arxiv`
    - `embedder_name`: String, sentence embedder name/tag in hugging-face, defaults to `paraphrase-MiniLM-L6-v2`
    - `nlp_name`: String, spacy model name/tag in hugging-face (if changed - needs to be spacy-installed prior), defaults to `en_core_sci_scibert`
    - `similarity_nlp_name`: String, spacy downstream trained model(for similarity) name/tag in hugging-face (if changed - needs to be spacy-installed prior), defaults to `en_core_sci_lg`
    - `kw_model_name`: String, keyword extraction model name/tag in hugging-face, defaults to `distilbert-base-nli-mean-tokens`
    - `high_gpu`: Bool, High GPU usage permitted, defaults to `False`
    - `refresh_models`: Bool, Refresh model downloads with given names (needs atleast one model name param above), defaults to False
    
    > during survey generation with `surveyor_obj.survey(query="my_research_query")`
    - `max_search`: int maximium number of papers to gaze at - defaults to `100`
    - `num_papers`: int maximium number of papers to download and analyse - defaults to `25`
    
### Extra Methods: 

```
*[Tip]* :(models can be changed in defaults or passed on during init along with `refresh-models=True`)
```

- `abstractive_summary` - takes a long text document (`string`) and returns a 1-paragraph abstract or “abstractive” summary (`string`)

	Input: 		
		
		`longtext` : string
		
	Returns: 		
		
		`summary` : string

- `extractive_summary` - takes a long text document (`string`) and returns a 1-paragraph of extracted highlights or “extractive” summary (`string`)

	Input: 		
		
		`longtext` : string
		
	Returns: 		
		
		`summary` : string

- `generate_title` - takes a long text document (`string`) and returns a generated title (`string`)

	Input: 		
		
		`longtext` : string
		
	Returns: 		
		
		`title` : string

- `extractive_highlights` - takes a long text document (`string`) and returns a list of extracted highlights (`[string]`), a list of keywords (`[string]`) and key phrases (`[string]`)

	Input: 		
		
		`longtext` : string
		
	Returns: 		
		
		`highlights` : [string]
		`keywords` : [string]
		`keyphrases` : [string]

- `extract_images_from_file` - takes a pdf file name (`string`) and returns a list of image filenames (`[string]`).

	Input: 		
		
		`pdf_file` : string
		
	Returns: 		
		
		`images_files` : [string]

- `extract_tables_from_file` - takes a pdf file name (`string`) and returns a list of csv filenames (`[string]`).

	Input: 		
		
		`pdf_file` : string
		
	Returns: 		
		
		`images_files` : [string]

- `cluster_lines` - takes a list of lines (`string`) and returns the topic-clustered sections (`dict(generated_title: [cluster_abstract])`) and clustered lines (`dict(cluster_id: [cluster_lines])`)

	Input: 		
		
		`lines` : [string]
		
	Returns: 		
		
		`sections` : dict(generated_title: [cluster_abstract])
		`clusters` : dict(cluster_id: [cluster_lines])

- `extract_headings` - *[for scientific texts - Assumes an ‘abstract’ heading present]* takes a text file name (`string`) and returns a list of headings (`[string]`) and refined lines (`[string]`). 
    
    `[Tip 1]` : Use `extract_sections` as a wrapper (e.g. `extract_sections(extract_headings(“/path/to/textfile”)`) to get heading-wise sectioned text with refined lines instead (`dict( heading: text)`)
    
    `[Tip 2]` : write the word ‘abstract’ at the start of the file text to get an extraction for non-scientific texts as well !!

	Input: 		
		
		`text_file` : string 		
		
	Returns: 
		
		`refined` : [string], 
		`headings` : [string]
		`sectioned_doc` : dict( heading: text) (Optional - Wrapper case)
