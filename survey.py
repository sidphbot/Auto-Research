from src.Surveyor import Surveyor

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate a survey just from a query !!')
    parser.add_argument('query', metavar='query_string', type=str,
                        help='your research query/keywords')
    parser.add_argument('--max_search', metavar='max_metadata_papers', type=int, default=None,
                        help='maximium number of papers to gaze at - defaults to 100')
    parser.add_argument('--num_papers', metavar='max_num_papers', type=int, default=None,
                        help='maximium number of papers to download and analyse - defaults to 25')
    parser.add_argument('--pdf_dir', metavar='pdf_dir', type=str, default=None,
                        help='pdf paper storage directory - defaults to arxiv_data/tarpdfs/')
    parser.add_argument('--txt_dir', metavar='txt_dir', type=str, default=None,
                        help='text-converted paper storage directory - defaults to arxiv_data/fulltext/')
    parser.add_argument('--img_dir', metavar='img_dir', type=str, default=None,
                        help='image storage directory - defaults to arxiv_data/images/')
    parser.add_argument('--tab_dir', metavar='tab_dir', type=str, default=None,
                        help='tables storage directory - defaults to arxiv_data/tables/')
    parser.add_argument('--dump_dir', metavar='dump_dir', type=str, default=None,
                        help='all_output_dir - defaults to arxiv_dumps/')
    parser.add_argument('--models_dir', metavar='save_models_dir', type=str, default=None,
                        help='directory to save models (> 5GB) - defaults to saved_models/')
    parser.add_argument('--title_model_name', metavar='title_model_name', type=str, default=None,
                        help='title model name/tag in hugging-face, defaults to \'Callidior/bert2bert-base-arxiv-titlegen\'')
    parser.add_argument('--ex_summ_model_name', metavar='extractive_summ_model_name', type=str, default=None,
                        help='extractive summary model name/tag in hugging-face, defaults to \'allenai/scibert_scivocab_uncased\'')
    parser.add_argument('--ledmodel_name', metavar='ledmodel_name', type=str, default=None,
                        help='led model(for abstractive summary) name/tag in hugging-face, defaults to \'allenai/led-large-16384-arxiv\'')
    parser.add_argument('--embedder_name', metavar='sentence_embedder_name', type=str, default=None,
                        help='sentence embedder name/tag in hugging-face, defaults to \'paraphrase-MiniLM-L6-v2\'')
    parser.add_argument('--nlp_name', metavar='spacy_model_name', type=str, default=None,
                        help='spacy model name/tag in hugging-face (if changed - needs to be spacy-installed prior), defaults to \'en_core_sci_scibert\'')
    parser.add_argument('--similarity_nlp_name', metavar='similarity_nlp_name', type=str, default=None,
                        help='spacy downstream model(for similarity) name/tag in hugging-face (if changed - needs to be spacy-installed prior), defaults to \'en_core_sci_lg\'')
    parser.add_argument('--kw_model_name', metavar='kw_model_name', type=str, default=None,
                        help='keyword extraction model name/tag in hugging-face, defaults to \'distilbert-base-nli-mean-tokens\'')
    parser.add_argument('--refresh_models', metavar='refresh_models', type=str, default=None,
                        help='Refresh model downloads with given names (needs atleast one model name param above), defaults to False')
    parser.add_argument('--high_gpu', metavar='high_gpu', type=str, default=None,
                        help='High GPU usage permitted, defaults to False')

    args = parser.parse_args()

    surveyor = Surveyor(
        pdf_dir=args.pdf_dir,
        txt_dir=args.txt_dir,
        img_dir=args.img_dir,
        tab_dir=args.tab_dir,
        dump_dir=args.dump_dir,
        models_dir=args.models_dir,
        title_model_name=args.title_model_name,
        ex_summ_model_name=args.ex_summ_model_name,
        ledmodel_name=args.ledmodel_name,
        embedder_name=args.embedder_name,
        nlp_name=args.nlp_name,
        similarity_nlp_name=args.similarity_nlp_name,
        kw_model_name=args.kw_model_name,
        refresh_models=args.refresh_models,
        high_gpu=args.high_gpu

    )

    surveyor.survey(args.query, max_search=args.max_search, num_papers=args.num_papers,
                                              debug=False, weigh_authors=False)

