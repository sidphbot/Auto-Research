# defaults for arxiv
DEFAULTS = {
    "max_search": 100,
    "num_papers": 20,
    "high_gpu": False,
    "pdf_dir": "arxiv_data/tarpdfs/",
    "txt_dir": "arxiv_data/fulltext/",
    "img_dir": "arxiv_data/images/",
    "tab_dir": "arxiv_data/tables/",
    "dump_dir": "arxiv_dumps/",
    "title_model_name": "Callidior/bert2bert-base-arxiv-titlegen",
    "ex_summ_model_name": "allenai/scibert_scivocab_uncased",
    "ledmodel_name": "allenai/led-large-16384-arxiv",
    "embedder_name": "paraphrase-MiniLM-L6-v2",
    "nlp_name": "en_core_sci_scibert",
    "similarity_nlp_name": "en_core_sci_lg",
    "kw_model_name": "distilbert-base-nli-mean-tokens",

}