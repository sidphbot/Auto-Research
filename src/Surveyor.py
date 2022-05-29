from pathlib import Path
from arxiv_public_data.fulltext import convert_directory_parallel
from arxiv_public_data import internal_citations
import torch
import os
from summarizer import Summarizer
from sentence_transformers import SentenceTransformer
import spacy
import numpy as np
from keybert import KeyBERT
import shutil, joblib
from distutils.dir_util import copy_tree

try:
    from transformers import *
except:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, AutoModel, LEDTokenizer, \
        LEDForConditionalGeneration

from src.defaults import DEFAULTS_CPU_COMPAT, DEFAULTS_HIGH_GPU



class Surveyor:
    '''
    A class to abstract all nlp and data mining helper functions as well as workflows
    required to generate the survey from a single query, with absolute configurability
    '''


    def __init__(
            self,
            pdf_dir=None,
            txt_dir=None,
            img_dir=None,
            tab_dir=None,
            dump_dir=None,
            models_dir=None,
            title_model_name=None,
            ex_summ_model_name=None,
            ledmodel_name=None,
            embedder_name=None,
            nlp_name=None,
            similarity_nlp_name=None,
            kw_model_name=None,
            high_gpu=False,
            refresh_models=False,
            no_save_models=False,
            print_fn=None,
            survey_print_fn=None
    ):
        '''
        Initializes models and directory structure for the surveyor

        Optional Params:
            - pdf_dir: String, pdf paper storage directory - defaults to arxiv_data/tarpdfs/
            - txt_dir: String, text-converted paper storage directory - defaults to arxiv_data/fulltext/
            - img_dir: String, image image storage directory - defaults to arxiv_data/images/
            - tab_dir: String, tables storage directory - defaults to arxiv_data/tables/
            - dump_dir: String, all_output_dir - defaults to arxiv_dumps/
            - models_dir: String, directory to save to huge models
            - title_model_name: String, title model name/tag in hugging-face, defaults to `Callidior/bert2bert-base-arxiv-titlegen`
            - ex_summ_model_name: String, extractive summary model name/tag in hugging-face, defaults to `allenai/scibert_scivocab_uncased`
            - ledmodel_name: String, led model(for abstractive summary) name/tag in hugging-face, defaults to `allenai/led-large-16384-arxiv`
            - embedder_name: String, sentence embedder name/tag in hugging-face, defaults to `paraphrase-MiniLM-L6-v2`
            - nlp_name: String, spacy model name/tag in hugging-face (if changed - needs to be spacy-installed prior), defaults to `en_core_sci_scibert`
            - similarity_nlp_name: String, spacy downstream trained model(for similarity) name/tag in hugging-face (if changed - needs to be spacy-installed prior), defaults to `en_core_sci_lg`
            - kw_model_name: String, keyword extraction model name/tag in hugging-face, defaults to `distilbert-base-nli-mean-tokens`
            - high_gpu: Bool, High GPU usage permitted, defaults to False
            - refresh_models: Bool, Refresh model downloads with given names (needs atleast one model name param above), defaults to False
            - no_save_models: forces refresh models
            
            - max_search: int maximium number of papers to gaze at - defaults to 100
            - num_papers: int maximium number of papers to download and analyse - defaults to 25

        '''
        self.print_fn = print
        if print_fn is not None:
            self.print_fn = print_fn
        self.survey_print_fn = self.print_fn
        if survey_print_fn is not None:
            self.survey_print_fn = survey_print_fn

        self.torch_device = 'cpu'
        self.print_fn("\nTorch_device: " + self.torch_device)
        if torch.cuda.is_available():
            self.torch_device = 'cuda'
            spacy.require_gpu()

        self.high_gpu = high_gpu
        self.DEFAULTS = DEFAULTS_CPU_COMPAT
        if self.high_gpu:
            self.DEFAULTS = DEFAULTS_HIGH_GPU

        if not kw_model_name:
            kw_model_name = self.DEFAULTS["kw_model_name"]
        self.num_papers = self.DEFAULTS['num_papers']
        self.max_search = self.DEFAULTS['max_search']
        if not models_dir:
            models_dir = self.DEFAULTS['models_dir']

        models_found = False
        if os.path.exists(models_dir) and not no_save_models:
            if len(os.listdir(models_dir)) > 6:
                models_found = True

        if not title_model_name:
            title_model_name = self.DEFAULTS["title_model_name"]
        if not ex_summ_model_name:
            ex_summ_model_name = self.DEFAULTS["ex_summ_model_name"]
        if not ledmodel_name:
            ledmodel_name = self.DEFAULTS["ledmodel_name"]
        if not embedder_name:
            embedder_name = self.DEFAULTS["embedder_name"]
        if not nlp_name:
            nlp_name = self.DEFAULTS["nlp_name"]
        if not similarity_nlp_name:
            similarity_nlp_name = self.DEFAULTS["similarity_nlp_name"]

        if refresh_models or not models_found:
            self.print_fn(f'\nInitializing models {"and saving (about 5GB)" if not no_save_models else ""}')
            if not no_save_models:
                self.clean_dirs([models_dir])

            self.title_tokenizer = AutoTokenizer.from_pretrained(title_model_name, trust_remote_code=True)
            self.title_model = AutoModelForSeq2SeqLM.from_pretrained(title_model_name, trust_remote_code=True).to(self.torch_device)
            self.title_model.eval()
            if not no_save_models:
                self.title_model.save_pretrained(models_dir + "/title_model")
            #self.title_tokenizer.save_pretrained(models_dir + "/title_tokenizer")

            # summary model
            self.custom_config = AutoConfig.from_pretrained(ex_summ_model_name)
            self.custom_config.output_hidden_states = True
            self.summ_tokenizer = AutoTokenizer.from_pretrained(ex_summ_model_name)
            self.summ_model = AutoModel.from_pretrained(ex_summ_model_name, config=self.custom_config).to(
                self.torch_device)
            self.summ_model.eval()
            if not no_save_models:
                self.summ_model.save_pretrained(models_dir + "/summ_model")
            #self.summ_tokenizer.save_pretrained(models_dir + "/summ_tokenizer")
            self.model = Summarizer(custom_model=self.summ_model, custom_tokenizer=self.summ_tokenizer)
            
            if 'led' in ledmodel_name:
                self.ledtokenizer = LEDTokenizer.from_pretrained(ledmodel_name)
                self.ledmodel = LEDForConditionalGeneration.from_pretrained(ledmodel_name).to(self.torch_device)
            elif 't5' in ledmodel_name:
                self.ledtokenizer = AutoTokenizer.from_pretrained(ledmodel_name)
                self.ledmodel = T5ForConditionalGeneration.from_pretrained(ledmodel_name).to(self.torch_device)
            elif 'bart' in ledmodel_name:
                self.ledtokenizer = AutoTokenizer.from_pretrained(ledmodel_name)
                self.ledmodel = BartForConditionalGeneration.from_pretrained(ledmodel_name).to(self.torch_device)
            self.ledmodel.eval()
            if not no_save_models:
                self.ledmodel.save_pretrained(models_dir + "/ledmodel")
            #self.ledtokenizer.save_pretrained(models_dir + "/ledtokenizer")

            self.embedder = SentenceTransformer(embedder_name)
            self.embedder.eval()
            if not no_save_models:
                self.embedder.save(models_dir + "/embedder")
        else:
            self.print_fn("\n- Initializing from previously saved models at" + models_dir)
            self.title_tokenizer = AutoTokenizer.from_pretrained(title_model_name)
            self.title_model = AutoModelForSeq2SeqLM.from_pretrained(models_dir + "/title_model").to(self.torch_device)
            self.title_model.eval()

            # summary model
            #self.summ_config = AutoConfig.from_pretrained(ex_summ_model_name)
            #self.summ_config.output_hidden_states = True
            self.summ_tokenizer = AutoTokenizer.from_pretrained(ex_summ_model_name)
            self.summ_model = AutoModel.from_pretrained(models_dir + "/summ_model").to(
                self.torch_device)
            self.summ_model.eval()
            self.model = Summarizer(custom_model=self.summ_model, custom_tokenizer=self.summ_tokenizer)

            if 'led' in ledmodel_name:
                self.ledtokenizer = LEDTokenizer.from_pretrained(ledmodel_name)
                self.ledmodel = LEDForConditionalGeneration.from_pretrained(models_dir + "/ledmodel").to(self.torch_device)
            elif 't5' in ledmodel_name:
                self.ledtokenizer = AutoTokenizer.from_pretrained(ledmodel_name)
                self.ledmodel = T5ForConditionalGeneration.from_pretrained(models_dir + "/ledmodel").to(self.torch_device)
            elif 'bart' in ledmodel_name:
                self.ledtokenizer = AutoTokenizer.from_pretrained(ledmodel_name)
                self.ledmodel = BartForConditionalGeneration.from_pretrained(models_dir + "/ledmodel").to(self.torch_device)
            self.ledmodel.eval()

            self.embedder = SentenceTransformer(models_dir + "/embedder")
            self.embedder.eval()

        self.nlp = spacy.load(nlp_name)
        self.similarity_nlp = spacy.load(similarity_nlp_name)
        self.kw_model = KeyBERT(kw_model_name)

        self.define_structure(pdf_dir=pdf_dir, txt_dir=txt_dir, img_dir=img_dir, tab_dir=tab_dir, dump_dir=dump_dir)

    def define_structure(self, pdf_dir=None, txt_dir=None, img_dir=None, tab_dir=None, dump_dir=None):

        if pdf_dir:
            self.pdf_dir = pdf_dir
        else:
            self.pdf_dir = self.DEFAULTS["pdf_dir"]

        if txt_dir:
            self.txt_dir = txt_dir
        else:
            self.txt_dir = self.DEFAULTS["txt_dir"]

        if img_dir:
            self.img_dir = img_dir
        else:
            self.img_dir = self.DEFAULTS["img_dir"]

        if tab_dir:
            self.tab_dir = tab_dir
        else:
            self.tab_dir = self.DEFAULTS["tab_dir"]

        if dump_dir:
            self.dump_dir = dump_dir
        else:
            self.dump_dir = self.DEFAULTS["dump_dir"]

        dirs = [self.pdf_dir, self.txt_dir, self.img_dir, self.tab_dir, self.dump_dir]
        if sum([True for dir in dirs if 'arxiv_data/' in dir]):
            base = os.path.dirname("arxiv_data/")
            if not os.path.exists(base):
                os.mkdir(base)
        self.clean_dirs(dirs)

    def clean_dirs(self, dirs):
        import shutil
        for d in dirs:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.mkdir(d)

    def pdf_route(self, pdf_dir, txt_dir, img_dir, tab_dir, dump_dir, papers_meta):
        ## Data prep

        import joblib
        # test full again - check images - check dfs !!

        self.clean_dirs([pdf_dir, txt_dir, img_dir, tab_dir, dump_dir])

        papers = papers_meta[:self.num_papers]
        selected_papers = papers
        ids_none, papers, cites = self.fetch_papers(dump_dir, img_dir, papers, pdf_dir, tab_dir, txt_dir)
        self.print_fn("\n- First stage paper collection complete, papers collected: \n" + ', '.join([p['id'] for p in papers]))
        new_papers = papers_meta[self.num_papers : self.num_papers + len(ids_none)]
        # _ = self.get_freq_cited(cites)
        '''
        filtered_idlist = []
        for c in self.get_freq_cited(cites):
            if c in 
        _, new_searched_papers = self.search(filtered_idlist)
        new_papers.extend(new_searched_papers)
        '''
        selected_papers.extend(new_papers)
        _, new_papers, _ = self.fetch_papers(dump_dir, img_dir, new_papers, pdf_dir, tab_dir, txt_dir, repeat=True)
        self.print_fn("\n- Second stage paper collection complete, new papers collected: \n" + ', '.join([p['id'] for p in new_papers]))
        papers.extend(new_papers)

        joblib.dump(papers, dump_dir + 'papers_extracted_pdf_route.dmp')
        copy_tree(img_dir, dump_dir + os.path.basename(img_dir))
        copy_tree(tab_dir, dump_dir + os.path.basename(tab_dir))

        self.print_fn("\n- Extracting section-wise highlights.. ")
        papers = self.extract_highlights(papers)

        return papers, selected_papers, cites


    def get_freq_cited(self, cites_dict, k=5):
        cites_list = []
        for k, v in cites_dict.items():
            cites_list.append(k)
            [cites_list.append(val) for val in v]
        cite_freqs = {cite: cites_list.count(cite) for cite in set(cites_list)}
        sorted_cites = dict(sorted(cite_freqs.items(), key=lambda item: item[1], reverse=True)[:5])

        return sorted_cites.keys()


    def fetch_papers(self, dump_dir, img_dir, papers, pdf_dir, tab_dir, txt_dir, repeat=False):
        import tempfile

        if repeat:
            with tempfile.TemporaryDirectory() as dirpath:
                self.print_fn("\n- downloading extra pdfs.. ")
                # full text preparation of selected papers
                self.download_pdfs(papers, dirpath)
                dirpath_pdfs = os.listdir(dirpath)
                for file_name in dirpath_pdfs:
                    full_file_name = os.path.join(dirpath, file_name)
                    if os.path.isfile(full_file_name):
                        shutil.copy(full_file_name, pdf_dir)
                self.print_fn("\n- converting extra pdfs.. ")
                self.convert_pdfs(dirpath, txt_dir)
        else:
            self.print_fn("\n- downloading pdfs.. ")
            # full text preparation of selected papers
            self.download_pdfs(papers, pdf_dir)
            self.print_fn("\n- converting pdfs.. ")
            self.convert_pdfs(pdf_dir, txt_dir)
        # plugging citations to our papers object
        self.print_fn("\n- plugging in citation network.. ")
        papers, cites = self.cocitation_network(papers, txt_dir)
        joblib.dump(papers, dump_dir + 'papers_selected_pdf_route.dmp')
        from distutils.dir_util import copy_tree
        copy_tree(txt_dir, dump_dir + os.path.basename(txt_dir))
        copy_tree(pdf_dir, dump_dir + os.path.basename(pdf_dir))
        self.print_fn("\n- extracting structure.. ")
        papers, ids_none = self.extract_structure(papers, pdf_dir, txt_dir, img_dir, dump_dir, tab_dir)
        return ids_none, papers, cites

    def tar_route(self, pdf_dir, txt_dir, img_dir, tab_dir, papers):
        ## Data prep

        import joblib
        # test full again - check images - check dfs !!

        self.clean_dirs([pdf_dir, txt_dir, img_dir, tab_dir])

        # full text preparation of selected papers
        self.download_sources(papers, pdf_dir)
        self.convert_pdfs(pdf_dir, txt_dir)

        # plugging citations to our papers object
        papers, cites = self.cocitation_network(papers, txt_dir)

        joblib.dump(papers, 'papers_selected_tar_route.dmp')

        papers = self.extract_structure(papers, pdf_dir, txt_dir, img_dir, tab_dir)

        joblib.dump(papers, 'papers_extracted_tar_route.dmp')

        return papers

    def build_doc(self, research_sections, papers, query=None, filename='survey.txt'):

        import arxiv2bib
        self.print_fn("\n- building bibliography entries.. ")
        bibentries = arxiv2bib.arxiv2bib([p['id'] for p in papers])
        bibentries = [r.bibtex() for r in bibentries]

        self.print_fn("\n- building final survey file .. at "+ filename)
        file = open(filename, 'w+')
        if query is None:
            query = 'Internal(existing) research'
        self.survey_print_fn("#### Generated_survey:")
        file.write("----------------------------------------------------------------------")
        file.write("Title: A survey on " + query)
        self.survey_print_fn("")
        self.survey_print_fn("----------------------------------------------------------------------")
        self.survey_print_fn("Title: A survey on " + query)
        file.write("Author: Auto-Research (github.com/sidphbot/Auto-Research)")
        self.survey_print_fn("Author: Auto-Research (github.com/sidphbot/Auto-Research)")
        file.write("Dev: Auto-Research (github.com/sidphbot/Auto-Research)")
        self.survey_print_fn("Dev: Auto-Research (github.com/sidphbot/Auto-Research)")
        file.write("Disclaimer: This survey is intended to be a research starter. This Survey is Machine-Summarized, "+
                   "\nhence some sentences might be wrangled or grammatically incorrect. However all sentences are "+
                   "\nmined with proper citations. As All of the text is practically quoted texted, hence to "+
                   "\nimprove visibility, all the papers are duly cited in the Bibiliography section. as bibtex "+
                   "\nentries(only to avoid LaTex overhead). ")
        self.survey_print_fn("Disclaimer: This survey is intended to be a research starter. This Survey is Machine-Summarized, "+
                "\nhence some sentences might be wrangled or grammatically incorrect. However all sentences are "+
                "\nmined with proper citations. As All of the text is practically quoted texted, hence to "+
                "\nimprove visibility, all the papers are duly cited in the Bibiliography section. as bibtex "+
                "\nentries(only to avoid LaTex overhead). ")
        file.write("----------------------------------------------------------------------")
        self.survey_print_fn("----------------------------------------------------------------------")
        file.write("")
        self.survey_print_fn("")
        file.write('ABSTRACT')
        self.survey_print_fn('ABSTRACT')
        self.survey_print_fn("=================================================")
        file.write("=================================================")
        file.write("")
        self.survey_print_fn("")
        file.write(research_sections['abstract'])
        self.survey_print_fn(research_sections['abstract'])
        file.write("")
        self.survey_print_fn("")
        file.write('INTRODUCTION')
        self.survey_print_fn('INTRODUCTION')
        self.survey_print_fn("=================================================")
        file.write("=================================================")
        file.write("")
        self.survey_print_fn("")
        file.write(research_sections['introduction'])
        self.survey_print_fn(research_sections['introduction'])
        file.write("")
        self.survey_print_fn("")
        for k, v in research_sections.items():
            if k not in ['abstract', 'introduction', 'conclusion']:
                file.write(k.upper())
                self.survey_print_fn(k.upper())
                self.survey_print_fn("=================================================")
                file.write("=================================================")
                file.write("")
                self.survey_print_fn("")
                file.write(v)
                self.survey_print_fn(v)
                file.write("")
                self.survey_print_fn("")
        file.write('CONCLUSION')
        self.survey_print_fn('CONCLUSION')
        self.survey_print_fn("=================================================")
        file.write("=================================================")
        file.write("")
        self.survey_print_fn("")
        file.write(research_sections['conclusion'])
        self.survey_print_fn(research_sections['conclusion'])
        file.write("")
        self.survey_print_fn("")

        file.write('REFERENCES')
        self.survey_print_fn('REFERENCES')
        self.survey_print_fn("=================================================")
        file.write("=================================================")
        file.write("")
        self.survey_print_fn("")
        for entry in bibentries:
            file.write(entry)
            self.survey_print_fn(entry)
            file.write("")
            self.survey_print_fn("")
        self.survey_print_fn("========================XXX=========================")
        file.write("========================XXX=========================")
        file.close()

    def build_basic_blocks(self, corpus_known_sections, corpus):

        research_blocks = {}
        for head, textarr in corpus_known_sections.items():
            if 'cuda' in self.torch_device:
                torch.cuda.empty_cache()
            # self.print_fn(head.upper())
            with torch.no_grad():
                summtext = self.model(" ".join([l.lower() for l in textarr]), ratio=0.5)
            res = self.nlp(summtext)
            res = set([str(sent) for sent in list(res.sents)])
            summtext = ''.join([line for line in res])
            # pself.print_fn(summtext)
            research_blocks[head] = summtext

        return research_blocks

    def abstractive_summary(self, longtext):
        '''
        faulty method
        input_ids = ledtokenizer(longtext, return_tensors="pt").input_ids
        global_attention_mask = torch.zeros_like(input_ids)
        # set global_attention_mask on first token
        global_attention_mask[:, 0] = 1

        sequences = ledmodel.generate(input_ids, global_attention_mask=global_attention_mask).sequences
        summary = ledtokenizer.batch_decode(sequences)
        '''
        if 'cuda' in self.torch_device:
                torch.cuda.empty_cache()
        inputs = self.ledtokenizer.prepare_seq2seq_batch(longtext, truncation=True, padding='longest',
                                                         return_tensors='pt').to(self.torch_device)
        with torch.no_grad():
            summary_ids = self.ledmodel.generate(**inputs)
        summary = self.ledtokenizer.batch_decode(summary_ids, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=True)
        res = self.nlp(summary[0])
        res = set([str(sent) for sent in list(res.sents)])
        summtext = ''.join([line for line in res])
        #self.print_fn("abstractive summary type:" + str(type(summary)))
        return summtext

    def get_abstract(self, abs_lines, corpus_known_sections, research_blocks):

        # abs_lines = " ".join(abs_lines)
        abs_lines = ""
        abs_lines += " ".join([l.lower() for l in corpus_known_sections['abstract']])
        abs_lines += research_blocks['abstract']
        # self.print_fn(abs_lines)

        try:
            return self.abstractive_summary(abs_lines)
        except:
            highlights = self.extractive_summary(abs_lines)
            return self.abstractive_summary(highlights)

    def get_corpus_lines(self, corpus):
        abs_lines = []
        types = set()
        for k, v in corpus.items():
            # self.print_fn(v)
            types.add(type(v))
            abstext = k + '. ' + v.replace('\n', ' ')
            abstext = self.nlp(abstext)
            abs_lines.extend([str(sent).lower() for sent in list(abstext.sents)])
        #self.print_fn("unique corpus value types:" + str(types))
        # abs_lines = '\n'.join([str(sent) for sent in abs_lines.sents])
        return abs_lines

    def get_sectioned_docs(self, papers, papers_meta):
        import random
        docs = []
        for p in papers:
            for section in p['sections']:
                if len(section['highlights']) > 0:
                    content = self.extractive_summary(''.join(section['highlights']))
                    docs.append(content)
        selected_pids = [p['id'] for p in papers]
        meta_abs = []
        for p in papers_meta:
            if p['id'] not in selected_pids:
                meta_abs.append(self.generate_title(p['abstract']))
        docs.extend(meta_abs)
        #self.print_fn("meta_abs num"+str(len(meta_abs)))
        #self.print_fn("selected_pids num"+str(len(selected_pids)))
        #self.print_fn("papers_meta num"+str(len(papers_meta)))
        #assert (len(meta_abs) + len(selected_pids) == len(papers_meta))
        assert ('str' in str(type(random.sample(docs, 1)[0])))
        return [doc for doc in docs if doc != '']


    def cluster_lines(self, abs_lines):
        from sklearn.cluster import KMeans
        # from bertopic import BERTopic
        # topic_model = BERTopic(embedding_model=embedder)
        if 'cuda' in self.torch_device:
                torch.cuda.empty_cache()
        corpus_embeddings = self.embedder.encode(abs_lines)
        # Normalize the embeddings to unit length
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        with torch.no_grad():
            optimal_k = self.model.calculate_optimal_k(' '.join(abs_lines), k_max=10)
        # Perform kmean clustering

        clustering_model = KMeans(n_clusters=optimal_k, n_init=20)
        # clustering_model = AgglomerativeClustering(n_clusters=optimal_k, affinity='cosine', linkage='average') #, affinity='cosine', linkage='average', distance_threshold=0.4)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        clustered_sentences = {}
        dummy_count = 0
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            if cluster_id not in clustered_sentences:
                clustered_sentences[cluster_id] = []
            '''
            if dummy_count < 5:
                self.print_fn("abs_line: "+abs_lines[sentence_id])
                self.print_fn("cluster_ID: "+str(cluster_id))
                self.print_fn("embedding: "+str(corpus_embeddings[sentence_id]))
                dummy_count += 1
            '''
            clustered_sentences[cluster_id].append(abs_lines[sentence_id])

        # for i, cluster in clustered_sentences.items():
        # self.print_fn("Cluster ", i+1)
        # self.print_fn(cluster)
        # self.print_fn("")

        return self.get_clustered_sections(clustered_sentences), clustered_sentences


    def get_clusters(self, papers, papers_meta):
        from sklearn.cluster import KMeans
        # from bertopic import BERTopic
        # topic_model = BERTopic(embedding_model=embedder)
        if 'cuda' in self.torch_device:
                torch.cuda.empty_cache()
        abs_lines = self.get_sectioned_docs(papers, papers_meta)
        corpus_embeddings = self.embedder.encode(abs_lines)
        # Normalize the embeddings to unit length
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        with torch.no_grad():
            optimal_k = self.model.calculate_optimal_k(' '.join(abs_lines), k_max=10)
        # Perform kmean clustering

        clustering_model = KMeans(n_clusters=optimal_k, n_init=20)
        # clustering_model = AgglomerativeClustering(n_clusters=optimal_k, affinity='cosine', linkage='average') #, affinity='cosine', linkage='average', distance_threshold=0.4)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        clustered_sentences = {}
        dummy_count = 0
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            if cluster_id not in clustered_sentences:
                clustered_sentences[cluster_id] = []
            '''
            if dummy_count < 5:
                self.print_fn("abs_line: "+abs_lines[sentence_id])
                self.print_fn("cluster_ID: "+str(cluster_id))
                self.print_fn("embedding: "+str(corpus_embeddings[sentence_id]))
                dummy_count += 1
            '''
            clustered_sentences[cluster_id].append(abs_lines[sentence_id])

        # for i, cluster in clustered_sentences.items():
        # self.print_fn("Cluster ", i+1)
        # self.print_fn(cluster)
        # self.print_fn("")

        return self.get_clustered_sections(clustered_sentences), clustered_sentences

    def generate_title(self, longtext):
        if 'cuda' in self.torch_device:
                torch.cuda.empty_cache()

        inputs = self.title_tokenizer.prepare_seq2seq_batch(longtext, truncation=True, padding='longest',
                                                            return_tensors='pt').to(self.torch_device)
        with torch.no_grad():
            summary_ids = self.title_model.generate(**inputs)
        summary = self.title_tokenizer.batch_decode(summary_ids, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=True)

        return str(summary[0])

    def get_clustered_sections(self, clustered_lines):
        clusters_dict = {}
        for i, cluster in clustered_lines.items():
            # self.print_fn(cluster)
            try:
                clusters_dict[self.generate_title(str(" ".join(cluster)))] = self.abstractive_summary(
                    str(" ".join(cluster)).lower())
            except:
                clusters_dict[self.generate_title(str(" ".join(cluster)))] = self.abstractive_summary(
                    self.extractive_summary(str(" ".join(cluster)).lower()))

        return clusters_dict

    def get_intro(self, corpus_known_sections, research_blocks):
        intro_lines = ""
        intro_lines += str(" ".join([l.lower() for l in corpus_known_sections['introduction']])) + str(
            " ".join([l.lower() for l in corpus_known_sections['conclusion']]))
        intro_lines += research_blocks['introduction'] + research_blocks['conclusion']
        try:
            return self.abstractive_summary(intro_lines)
        except:
            return self.abstractive_summary(self.extractive_summary(intro_lines))

    def get_conclusion(self, research_sections):
        paper_body = ""
        for k, v in research_sections.items():
            paper_body += v

        try:
            return self.abstractive_summary(paper_body)
        except:
            return self.abstractive_summary(self.extractive_summary(paper_body))

    def build_corpus_sectionwise(self, papers):
        known = ['abstract', 'introduction', 'conclusion']
        corpus_known_sections = {}
        for kh in known:
            khtext = []
            for p in papers:
                for section in p['sections']:
                    if kh in section['heading']:
                        khtext.extend(section['highlights'])
                        # self.print_fn(khtext)
            corpus_known_sections[kh] = khtext
        return corpus_known_sections

    def standardize_headings(self, papers):
        known = ['abstract', 'introduction', 'discussion', 'relatedwork', 'contribution', 'analysis', 'experiments',
                 'conclusion']
        for p in papers:
            # self.print_fn("================================")
            headings = [section['heading'] for section in p['sections'] if len(section['heading'].split()) < 3]
            # self.print_fn("id: "+ str(p['id'])+"\nHeadings: \n"+str('\n'.join(headings)))
            for kh in known:
                for section in p['sections']:
                    if len(section['heading'].split()) < 3:
                        # self.print_fn(section['heading'])
                        if kh in ''.join(filter(str.isalpha, section['heading'].replace(' ', '').lower())):
                            # self.print_fn("orig head: "+ section['heading'] +", plain head:" + kh)
                            section['heading'] = kh
        return papers

    def build_corpus(self, papers, papers_meta):
        corpus = self.build_meta_corpus(papers_meta)
        for p in papers:
            ph = []
            for sid, section in enumerate(p['sections']):
                ph.extend(section['highlights'])
            for pid, ls in corpus.items():
                if pid == p['id']:
                    corpus[pid] = p['abstract'] + str(' '.join(ph))
        '''
        self.print_fn("==================    final corpus       ====================")
        self.print_fn('\n'.join([str("paper: "+ get_by_pid(pid, papers_meta)['title']+" \nhighlight count: " + str(len(phs))) for pid, phs in corpus.items()]))
        self.print_fn("========    sample point       ========")
        p = random.choice(list(papers))
        self.print_fn("paper: "+ p['title']+" \nhighlights: " + str(corpus[p['id']]))
        self.print_fn("========    sample meta point       ========")
        p = random.choice(list(papers_meta))
        self.print_fn("meta paper: "+ p['title']+" \nhighlights: " + str(corpus[p['id']]))
        '''
        return corpus

    def get_by_pid(self, pid, papers):
        for p in papers:
            if p['id'] == pid:
                return p

    def build_meta_corpus(self, papers):
        meta_corpus = {}
        for p in papers:
            # pself.print_fn(p)
            pid = p['id']
            ptext = p['title'] + ". " + p['abstract']
            doc = self.nlp(ptext)
            phs, _, _ = self.extractive_highlights([str(sent) for sent in list(doc.sents)])
            meta_corpus[pid] = str(' '.join(phs))
        '''
        self.print_fn("==================    meta corpus       ====================")
        self.print_fn('\n'.join([str("paper: "+ get_by_pid(pid, papers)['title']+" \nhighlight count: " + str(len(phs))) for pid, phs in meta_corpus.items()]))
        self.print_fn("========    sample point       ========")
        p = random.choice(list(papers))
        self.print_fn("paper: "+ p['title']+" \nhighlights: " + str(meta_corpus[p['id']]))
        '''
        return meta_corpus

    def select_papers(self, papers, query, num_papers=20):
        import numpy as np
        # self.print_fn("paper sample: ")
        # self.print_fn(papers)
        meta_corpus = self.build_meta_corpus(papers)
        scores = []
        pids = []
        for id, highlights in meta_corpus.items():
            score = self.text_para_similarity(query, highlights)
            scores.append(score)
            pids.append(id)
            # self.print_fn("corpus item: " + str(self.get_by_pid(id, papers)['title']))

        idx = np.argsort(scores)[:num_papers]
        #for i in range(len(scores)):
        #    self.print_fn("paper: " + str(self.get_by_pid(pids[i], papers)['title']))
        #    self.print_fn("score: " + str(scores[i]))
        # self.print_fn("argsort ids("+str(num_papers)+" papers): "+ str(idx))
        idx = [pids[i] for i in idx]
        # self.print_fn("argsort pids("+str(num_papers)+" papers): "+ str(idx))
        papers_selected = [p for p in papers if p['id'] in idx]
        # assert(len(papers_selected)==num_papers)
        self.print_fn("num papers selected: " + str(len(papers_selected)))
        for p in papers_selected:
            self.print_fn("Selected Paper: " + p['title'])

        #self.print_fn("constrast with natural selection: forward")
        #for p in papers[:4]:
        #    self.print_fn("Selected Paper: " + p['title'])
        #self.print_fn("constrast with natural selection: backward")
        #for p in papers[-4:]:
        #    self.print_fn("Selected Paper: " + p['title'])
        # arxiv search producing better relevnce
        return papers_selected

    def extractive_summary(self, text):
        if 'cuda' in self.torch_device:
                torch.cuda.empty_cache()
        with torch.no_grad():
            res = self.model(text, ratio=0.5)
        res_doc = self.nlp(res)
        return " ".join(set([str(sent) for sent in list(res_doc.sents)]))

    def extractive_highlights(self, lines):
        # text = " ".join(lines)
        # text_doc = nlp(" ".join([l.lower() for l in lines]))
        # text = ' '.join([ str(sent) for sent in list(text_doc.sents)])
        if 'cuda' in self.torch_device:
                torch.cuda.empty_cache()
        with torch.no_grad():
            res = self.model(" ".join([l.lower() for l in lines]), ratio=0.5, )
        res_doc = self.nlp(res)
        res_lines = set([str(sent) for sent in list(res_doc.sents)])
        # self.print_fn("\n- ".join(res_sents))
        with torch.no_grad():
            keywords = self.kw_model.extract_keywords(str(" ".join([l.lower() for l in lines])), stop_words='english')
            keyphrases = self.kw_model.extract_keywords(str(" ".join([l.lower() for l in lines])),
                                                    keyphrase_ngram_range=(4, 4),
                                                    stop_words='english', use_mmr=True, diversity=0.7)
        return res_lines, keywords, keyphrases

    def extract_highlights(self, papers):
        for p in papers:
            sid = 0
            p['sections'] = []
            for heading, lines in p['body_text'].items():
                hs, kws, kps = self.extractive_highlights(lines)
                p['sections'].append({
                    'sid': sid,
                    'heading': heading,
                    'text': lines,
                    'highlights': hs,
                    'keywords': kws,
                    'keyphrases': kps,
                })
                sid += 1
        return papers

    def extract_structure(self, papers, pdf_dir, txt_dir, img_dir, dump_dir, tab_dir, tables=False):
        self.print_fn("\n- extracting sections.. ")
        papers, ids_none = self.extract_parts(papers, txt_dir, dump_dir)

        self.print_fn("\n- extracting images.. for future correlation use-cases ")
        papers = self.extract_images(papers, pdf_dir, img_dir)

        if tables:
            self.print_fn("\n- extracting tables.. for future correlation use-cases ")
            papers = self.extract_tables(papers, pdf_dir, tab_dir)

        return papers, ids_none

    def extract_parts(self, papers, txt_dir, dump_dir):

        headings_all = {}
        # refined = []
        # model = build_summarizer()
        #for file in glob.glob(txt_dir + '/*.txt'):
        for p in papers:
            file = txt_dir + '/'+ p['id'] +'.txt'
            refined, headings_extracted = self.extract_headings(file)
            sections = self.extract_sections(headings_extracted, refined)
            # highlights = {k: extract_highlights(model,v) for k, v in sections.items()}
            #p = self.get_by_file(file, papers)
            #if len(headings_extracted) > 3:
            p['body_text'] = sections
            # p['body_highlights'] = highlights
            headings_all[p['id']] = headings_extracted

        ids_none = {i: h for i, h in headings_all.items() if len(h) < 3}

        '''
        for f, h in headings_all.items():
            if len(h) < 4:
                self.print_fn("=================headings almost undetected================")
                self.print_fn(f)
                self.print_fn(h)
        '''
        # from pprint import pprint
        # pself.print_fn({f: len(h) for f,h in headings_all.items()})
        papers_none = [p for p in papers if p['id'] in ids_none]
        for p in papers_none:
            os.remove(txt_dir + '/'+ p['id'] + '.txt')
            papers.remove(p)

        return papers, ids_none

    def check_para(self, df):
        size = 0
        for col in df.columns:
            size += df[col].apply(lambda x: len(str(x))).median()
        return size / len(df.columns) > 25

    def scan_blocks(self, lines):
        lines_mod = [line.strip().replace('\n', '') for line in lines if len(line.strip().replace('\n', '')) > 3]
        for i in range(len(lines_mod)):
            yield lines_mod[i:i + 3]

    def extract_sections(self, headings, lines, min_part_length=2):
        sections = {}
        self.check_list_elems_in_list(headings, lines)
        head_len = len(headings)
        for i in range(len(headings) - 1):
            start = headings[i]
            end = headings[i + 1]
            section = self.get_section(start, end, lines)
            # self.print_fn(start + " : "+ str(len(section)) +" lines")
            '''
            if i > 0:
              old = headings[i-1]
              if len(section) < min_part_length + 1:
                sections[old].extend(start)
                sections[old].extend(section)
              else:
                sections[start] = section
            else:
              sections[start] = section
            '''
            sections[start] = section
        return {k: v for k, v in sections.items()}

    def is_rubbish(self, s, rubbish_tolerance=0.2, min_char_len=4):
        # numbers = sum(c.isdigit() for c in s)
        letters = sum(c.isalpha() for c in s)
        spaces = sum(c.isspace() for c in s)
        # others  = len(s) - numbers - letters - spaces
        if len(s) == 0:
            return False
        if ((len(s) - (letters + spaces)) / len(s) >= rubbish_tolerance) or self.alpha_length(s) < min_char_len:
            return True
        else:
            return False

    def get_section(self, first, last, lines):
        try:
            assert (first in lines)
            assert (last in lines)
            # start = lines.index( first ) + len( first )
            # end = lines.index( last, start )
            start = [i for i in range(len(lines)) if first is lines[i]][0]
            end = [i for i in range(len(lines)) if last is lines[i]][0]
            section_lines = lines[start + 1:end]
            # self.print_fn("heading: " + str(first))
            # self.print_fn("section_lines: "+ str(section_lines))
            # self.print_fn(section_lines)
            return section_lines
        except ValueError:
            self.print_fn("value error :")
            self.print_fn("first heading :" + str(first) + ", second heading :" + str(last))
            self.print_fn("first index :" + str(start) + ", second index :" + str(end))
            return ""

    def check_list_elems_in_list(self, headings, lines):
        import numpy as np
        # [self.print_fn(head) for head in headings if head not in lines ]
        return np.all([True if head in lines else False for head in headings])

    def check_first_char_upper(self, text):
        for c in text:
            if c.isspace():
                continue
            elif c.isalpha():
                return c.isupper()

    def extract_headings(self, txt_file):
        import re

        fulltext = self.read_paper(txt_file)
        lines = self.clean_lines(fulltext)

        refined, headings = self.scan_text(lines)
        assert (self.check_list_elems_in_list(headings, refined))
        headings = self.check_duplicates(headings)

        # self.print_fn('===========================================')
        # self.print_fn(txt_file +": first scan: \n"+str(len(headings))+" headings")
        # self.print_fn('\n'.join(headings))

        # scan_failed - rescan with first match for abstract hook
        if len(headings) == 0:
            # self.print_fn('===================')
            # self.print_fn("run 1 failed")
            abs_cans = [line for line in lines if 'abstract' in re.sub("\s+", "", line.strip().lower())]
            if len(abs_cans) != 0:
                abs_head = abs_cans[0]
                refined, headings = self.scan_text(lines, abs_head=abs_head)
                self.check_list_elems_in_list(headings, refined)
                headings = self.check_duplicates(headings)
                # self.print_fn('===================')
                # self.print_fn(txt_file +": second scan: \n"+str(len(headings))+" headings")

        # if len(headings) == 0:
        # self.print_fn("heading scan failed completely")

        return refined, headings

    def check_duplicates(self, my_list):
        my_finallist = []
        dups = [s for s in my_list if my_list.count(s) > 1]
        if len(dups) > 0:
            [my_finallist.append(n) for n in my_list if n not in my_finallist]

        # self.print_fn("original: "+str(len(my_list))+" new: "+str(len(my_finallist)))
        return my_finallist

    def clean_lines(self, text):
        import numpy as np
        import re
        # doc = nlp(text)
        # lines = [str(sent) for sent in doc.sents]
        lines = text.replace('\r', '').split('\n')
        lines = [line for line in lines if not self.is_rubbish(line)]
        lines = [line for line in lines if
                 re.match("^[a-zA-Z1-9\.\[\]\(\):\-,\"\"\s]*$", line) and not 'Figure' in line and not 'Table' in line]

        lengths_cleaned = [self.alpha_length(line) for line in lines]
        mean_length_cleaned = np.median(lengths_cleaned)
        lines_standardized = []
        for line in lines:
            if len(line) >= (1.8 * mean_length_cleaned):
                first_half = line[0:len(line) // 2]
                second_half = line[len(line) // 2 if len(line) % 2 == 0 else ((len(line) // 2) + 1):]
                lines_standardized.append(first_half)
                lines_standardized.append(second_half)
            else:
                lines_standardized.append(line)

        return lines

    def scan_text(self, lines, abs_head=None):
        import re
        # self.print_fn('\n'.join(lines))
        record = False
        headings = []
        refined = []
        for i in range(1, len(lines) - 4):
            line = lines[i]
            line = line.replace('\n', '').strip()
            if 'abstract' in re.sub("\s+", "", line.strip().lower()) and len(line) - len('abstract') < 5 or (
                    abs_head is not None and abs_head in line):
                record = True
                headings.append(line)
                refined.append(line)
            if 'references' in re.sub("\s+", "", line.strip().lower()) and len(line) - len('references') < 5:
                headings.append(line)
                refined.append(line)
                break
            elif 'bibliography' in re.sub("\s+", "", line.strip().lower()) and len(line) - len('bibliography') < 5:
                headings.append(line)
                refined.append(line)
                break
            refined, headings = self.scanline(record, headings, refined, i, lines)
            # self.print_fn('=========in scan_text loop i : '+str(i)+' heading count : '+str(len(headings))+'  =========')
        return refined, headings

    def scanline(self, record, headings, refined, id, lines):
        import numpy as np
        import re
        line = lines[id]

        if not len(line) == 0:
            # self.print_fn("in scanline")
            # self.print_fn(line)
            if record:
                refined.append(line)
                if len(lines[id - 1]) == 0 or len(lines[id + 1]) == 0 or re.match(
                        "^[1-9XVIABCD]{0,4}(\.{0,1}[1-9XVIABCD]{0,4}){0, 3}\s{0,2}[A-Z][a-zA-Z\:\-\s]*$",
                        line) and self.char_length(line) > 7:
                    # self.print_fn("candidate")
                    # self.print_fn(line)
                    if np.mean([len(s) for s in lines[id + 2:id + 6]]) > 40 and self.check_first_char_upper(
                            line) and re.match("^[a-zA-Z1-9\.\:\-\s]*$", line) and len(line.split()) < 10:
                        # if len(line) < 20 and np.mean([len(s) for s in lines[i+1:i+5]]) > 30 :
                        headings.append(line)
                        assert (line in refined)
                        # self.print_fn("selected")
                        # self.print_fn(line)
                else:
                    known_headings = ['introduction', 'conclusion', 'abstract', 'references', 'bibliography']
                    missing = [h for h in known_headings if not np.any([True for head in headings if h in head])]
                    # for h in missing:
                    head = [line for h in missing if h in re.sub("\s+", "", line.strip().lower())]
                    # head = [line for known]
                    if len(head) > 0:
                        headings.append(head[0])
                        assert (head[0] in refined)

        return refined, headings

    def char_length(self, s):
        # numbers = sum(c.isdigit() for c in s)
        letters = sum(c.isalpha() for c in s)
        # spaces  = sum(c.isspace() for c in s)
        # others  = len(s) - numbers - letters - spaces
        return letters

    def get_by_file(self, file, papers):
        import os
        pid = os.path.basename(file)
        pid = pid.replace('.txt', '').replace('.pdf', '')
        for p in papers:
            if p['id'] == pid:
                return p
        self.print_fn("\n- paper not found by file, \nfile: "+file+"\nall papers: "+', '.join([p['id'] for p in papers]))


    def alpha_length(self, s):
        # numbers = sum(c.isdigit() for c in s)
        letters = sum(c.isalpha() for c in s)
        spaces = sum(c.isspace() for c in s)
        # others  = len(s) - numbers - letters - spaces
        return letters + spaces

    def check_append(self, baselist, addstr):
        check = False
        for e in baselist:
            if addstr in e:
                check = True
        if not check:
            baselist.append(addstr)
        return baselist

    def extract_images(self, papers, pdf_dir, img_dir):
        import fitz
        # self.print_fn("in images")
        for p in papers:
            file = pdf_dir + p['id'] + ".pdf"
            pdf_file = fitz.open(file)
            images = []
            for page_index in range(len(pdf_file)):
                page = pdf_file[page_index]
                images.extend(page.getImageList())
            images_files = [self.save_image(pdf_file.extractImage(img[0]), i, p['id'], img_dir) for i, img in
                            enumerate(set(images)) if img[0]]
            # self.print_fn(len(images_per_paper))
            p['images'] = images_files
            # self.print_fn(len(p.keys()))
        # self.print_fn(papers[0].keys())
        return papers


    def extract_images_from_file(self, pdf_file_name, img_dir):
        import fitz
        pdf_file = fitz.open(pdf_file_name)
        images = []
        for page_index in range(len(pdf_file)):
            page = pdf_file[page_index]
            images.extend(page.getImageList())
        images_files = [self.save_image(pdf_file.extractImage(img[0]), i, pdf_file_name.replace('.pdf', ''), img_dir) for i, img in
                        enumerate(set(images)) if img[0]]
        return images_files

    def save_image(self, base_image, img_index, pid, img_dir):
        from PIL import Image
        import io
        image_bytes = base_image["image"]
        # get the image extension
        image_ext = base_image["ext"]
        # load it to PIL
        image = Image.open(io.BytesIO(image_bytes))
        # save it to local disk
        fname = img_dir + "/" + str(pid) + "_" + str(img_index + 1) + "." + image_ext
        image.save(open(f"{fname}", "wb"))
        # self.print_fn(fname)
        return fname

    def save_tables(self, dfs, pid, tab_dir):
        # todo
        dfs = [df for df in dfs if not self.check_para(df)]
        files = []
        for df in dfs:
            filename = tab_dir + "/" + str(pid) + ".csv"
            files.append(filename)
            df.to_csv(filename, index=False)
        return files

    def extract_tables(self, papers, pdf_dir, tab_dir):
        import tabula
        check = True
        # for file in glob.glob(pdf_dir+'/*.pdf'):
        for p in papers:
            dfs = tabula.read_pdf(pdf_dir + p['id'] + ".pdf", pages='all', multiple_tables=True, silent=True)
            p['tables'] = self.save_tables(dfs, p['id'], tab_dir)
        # self.print_fn(papers[0].keys())
        return papers

    def extract_tables_from_file(self, pdf_file_name, tab_dir):
        import tabula
        check = True
        # for file in glob.glob(pdf_dir+'/*.pdf'):
        dfs = tabula.read_pdf(pdf_file_name, pages='all', multiple_tables=True, silent=True)

        return self.save_tables(dfs, pdf_file_name.replace('.pdf', ''), tab_dir)

    def search(self, query_text=None, id_list=None, max_search=100):
        import arxiv
        from urllib.parse import urlparse

        if query_text:
            search = arxiv.Search(
                query=query_text,
                max_results=max_search,
                sort_by=arxiv.SortCriterion.Relevance
            )
        else:
            id_list = [id for id in id_list if '.' in id]
            search = arxiv.Search(
                id_list=id_list
            )

        results = [result for result in search.get()]

        searched_papers = []
        discarded_ids = []
        for result in results:
            id = urlparse(result.entry_id).path.split('/')[-1].split('v')[0]
            if '.' in id:
                paper = {
                    'id': id,
                    'title': result.title,
                    'comments': result.comment if result.journal_ref else "None",
                    'journal-ref': result.journal_ref if result.journal_ref else "None",
                    'doi': str(result.doi),
                    'primary_category': result.primary_category,
                    'categories': result.categories,
                    'license': None,
                    'abstract': result.summary,
                    'published': result.published,
                    'pdf_url': result.pdf_url,
                    'links': [str(l) for l in result.links],
                    'update_date': result.updated,
                    'authors': [str(a.name) for a in result.authors],
                }
                searched_papers.append(paper)
            else:
                discarded_ids.append(urlparse(result.entry_id).path.split('/')[-1].split('v')[0])

        self.print_fn("\n- Papers discarded due to id error [arxiv api bug: #74] :\n" + str(discarded_ids))

        return results, searched_papers

    def download_pdfs(self, papers, pdf_dir):
        import arxiv
        from urllib.parse import urlparse
        ids = [p['id'] for p in papers]
        # asert(False)
        papers_filtered = arxiv.Search(id_list=ids).get()
        for p in papers_filtered:
            p_id = str(urlparse(p.entry_id).path.split('/')[-1]).split('v')[0]
            download_file = pdf_dir + "/" + p_id + ".pdf"
            p.download_pdf(filename=download_file)


    def download_sources(self, papers, src_dir):
        import arxiv
        from urllib.parse import urlparse
        ids = [p['id'] for p in papers]
        # asert(False)
        papers_filtered = arxiv.Search(id_list=ids).get()
        for p in papers_filtered:
            p_id = str(urlparse(p.entry_id).path.split('/')[-1]).split('v')[0]
            download_file = src_dir + "/" + p_id + ".tar.gz"
            p.download_source(filename=download_file)

    def convert_pdfs(self, pdf_dir, txt_dir):
        import glob, shutil

        import multiprocessing
        # import arxiv_public_data

        convert_directory_parallel(pdf_dir, multiprocessing.cpu_count())
        for file in glob.glob(pdf_dir + '/*.txt'):
            shutil.move(file, txt_dir)

    def read_paper(self, path):
        f = open(path, 'r', encoding="utf-8")
        text = str(f.read())
        f.close()
        return text

    def cocitation_network(self, papers, txt_dir):
        import multiprocessing

        cites = internal_citations.citation_list_parallel(N=multiprocessing.cpu_count(), directory=txt_dir)

        for p in papers:
            p['cites'] = cites[p['id']]
        return papers, cites

    def lookup_author(self, author_query):

        from scholarly import scholarly
        import operator
        # Retrieve the author's data, fill-in, and print
        self.print_fn("Searching Author: " + author_query)
        search_result = next(scholarly.search_author(author_query), None)

        if search_result is not None:
            author = scholarly.fill(search_result)
            author_stats = {
                'name': author_query,
                'affiliation': author['affiliation'] if author['affiliation'] else None,
                'citedby': author['citedby'] if 'citedby' in author.keys() else 0,
                'most_cited_year': max(author['cites_per_year'].items(), key=operator.itemgetter(1))[0] if len(
                    author['cites_per_year']) > 0 else None,
                'coauthors': [c['name'] for c in author['coauthors']],
                'hindex': author['hindex'],
                'impact': author['i10index'],
                'interests': author['interests'],
                'publications': [{'title': p['bib']['title'], 'citations': p['num_citations']} for p in
                                 author['publications']],
                'url_picture': author['url_picture'],
            }
        else:
            self.print_fn("author not found")
            author_stats = {
                'name': author_query,
                'affiliation': "",
                'citedby': 0,
                'most_cited_year': None,
                'coauthors': [],
                'hindex': 0,
                'impact': 0,
                'interests': [],
                'publications': [],
                'url_picture': "",
            }

        # pself.print_fn(author_stats)
        return author_stats

    def author_stats(self, papers):
        all_authors = []
        for p in papers:
            paper_authors = [a for a in p['authors']]
            all_authors.extend(paper_authors)

        searched_authors = [self.lookup_author(a) for a in set(all_authors)]

        return searched_authors

    def text_similarity(self, text1, text2):
        doc1 = self.similarity_nlp(text1)
        doc2 = self.similarity_nlp(text2)
        return doc1.similarity(doc2)

    def text_para_similarity(self, text, lines):
        doc1 = self.similarity_nlp(text)
        doc2 = self.similarity_nlp(" ".join(lines))
        return doc1.similarity(doc2)

    def para_para_similarity(self, lines1, lines2):
        doc1 = self.similarity_nlp(" ".join(lines1))
        doc2 = self.similarity_nlp(" ".join(lines2))
        return doc1.similarity(doc2)

    def text_image_similarity(self, text, image):
        pass

    def ask(self, corpus, question):
        text = " ".join(corpus)
        import torch
        inputs = self.qatokenizer(question, text, return_tensors='pt')
        start_positions = torch.tensor([1])
        end_positions = torch.tensor([3])
        outputs = self.qamodel(**inputs, start_positions=start_positions, end_positions=end_positions)
        self.print_fn("context: " + text)
        self.print_fn("question: " + question)
        self.print_fn("outputs: " + outputs)
        return outputs

    def zip_outputs(self, dump_dir, zip_name):
        import zipfile
        def zipdir(path, ziph):
            # ziph is zipfile handle
            for root, dirs, files in os.walk(path):
                for file in files:
                    ziph.write(os.path.join(root, file),
                               os.path.relpath(os.path.join(root, file),
                                               os.path.join(path, '../..')))

        
        zipf = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
        zipdir(dump_dir, zipf)

    def survey(self, query=None, id_list=None, max_search=None, num_papers=None, debug=False, weigh_authors=False):
        import joblib
        import os, shutil
        if not max_search:
            max_search = self.DEFAULTS['max_search']
        if not num_papers:
            num_papers = self.DEFAULTS['num_papers']
        if (query is None) and (id_list is None):
            raise ValueError('please provide a base to survey on: list of arxiv IDs or a few research keywords')
        # arxiv api relevance search and data preparation
        self.print_fn("\n- searching arXiv for top 100 papers.. ")
        results, searched_papers = self.search(query, id_list, max_search=max_search)
        joblib.dump(searched_papers, self.dump_dir + 'papers_metadata.dmp')
        self.print_fn("\n- found " + str(len(searched_papers)) + " papers")

        # paper selection by scibert vector embedding relevance scores
        # papers_selected = select_papers(searched_papers, query, num_papers=num_papers)

        papers_highlighted, papers_selected, cites = self.pdf_route(self.pdf_dir, self.txt_dir, self.img_dir, self.tab_dir, self.dump_dir,
                                            searched_papers)

        if weigh_authors:
            authors = self.author_stats(papers_highlighted)

        joblib.dump(papers_highlighted, self.dump_dir + 'papers_highlighted.dmp')

        self.print_fn("\n- Standardizing known section headings per paper.. ")
        papers_standardized = self.standardize_headings(papers_highlighted)
        joblib.dump(papers_standardized, self.dump_dir + 'papers_standardized.dmp')

        self.print_fn("\n- Building paper-wise corpus.. ")
        corpus = self.build_corpus(papers_highlighted, searched_papers)
        joblib.dump(corpus, self.dump_dir + 'corpus.dmp')

        self.print_fn("\n- Building section-wise corpus.. ")
        corpus_sectionwise = self.build_corpus_sectionwise(papers_standardized)
        joblib.dump(corpus_sectionwise, self.dump_dir + 'corpus_sectionwise.dmp')

        self.print_fn("\n- Building basic research highlights.. ")
        research_blocks = self.build_basic_blocks(corpus_sectionwise, corpus)
        joblib.dump(research_blocks, self.dump_dir + 'research_blocks.dmp')

        self.print_fn("\n- Reducing corpus to lines.. ")
        corpus_lines = self.get_corpus_lines(corpus)
        joblib.dump(corpus_lines, self.dump_dir + 'corpus_lines.dmp')

        # temp
        # searched_papers = joblib.load(dump_dir + 'papers_metadata.dmp')
        '''
        papers_highlighted = joblib.load(dump_dir + 'papers_highlighted.dmp')
        corpus = joblib.load(dump_dir + 'corpus.dmp')
        papers_standardized = joblib.load(dump_dir + 'papers_standardized.dmp')
        corpus_sectionwise = joblib.load(dump_dir + 'corpus_sectionwise.dmp')
        research_blocks = joblib.load(dump_dir + 'research_blocks.dmp')
        corpus_lines = joblib.load(dump_dir + 'corpus_lines.dmp')
        '''

        '''
        self.print_fn("papers_highlighted types:"+ str(np.unique([str(type(p['sections'][0]['highlights'])) for p in papers_highlighted])))
        self.print_fn("papers_highlighted example:")
        self.print_fn(random.sample(list(papers_highlighted), 1)[0]['sections'][0]['highlights'])
        self.print_fn("corpus types:"+ str(np.unique([str(type(txt)) for k,txt in corpus.items()])))
        self.print_fn("corpus example:")
        self.print_fn(random.sample(list(corpus.items()), 1)[0])
        self.print_fn("corpus_lines types:"+ str(np.unique([str(type(txt)) for txt in corpus_lines])))
        self.print_fn("corpus_lines example:")
        self.print_fn(random.sample(list(corpus_lines), 1)[0])
        self.print_fn("corpus_sectionwise types:"+ str(np.unique([str(type(txt)) for k,txt in corpus_sectionwise.items()])))
        self.print_fn("corpus_sectionwise example:")
        self.print_fn(random.sample(list(corpus_sectionwise.items()), 1)[0])
        self.print_fn("research_blocks types:"+ str(np.unique([str(type(txt)) for k,txt in research_blocks.items()])))
        self.print_fn("research_blocks example:")
        self.print_fn(random.sample(list(research_blocks.items()), 1)[0])
        '''
        # self.print_fn("corpus types:"+ str(np.unique([type(txt) for k,txt in corpus.items()])))

        self.print_fn("\n- Building abstract.. ")
        abstract_block = self.get_abstract(corpus_lines, corpus_sectionwise, research_blocks)
        joblib.dump(abstract_block, self.dump_dir + 'abstract_block.dmp')
        '''
        self.print_fn("abstract_block type:"+ str(type(abstract_block)))
        self.print_fn("abstract_block:")
        self.print_fn(abstract_block)
        '''

        self.print_fn("\n- Building introduction.. ")
        intro_block = self.get_intro(corpus_sectionwise, research_blocks)
        joblib.dump(intro_block, self.dump_dir + 'intro_block.dmp')
        '''
        self.print_fn("intro_block type:"+ str(type(intro_block)))
        self.print_fn("intro_block:")
        self.print_fn(intro_block)
        '''
        self.print_fn("\n- Building custom sections.. ")
        clustered_sections, clustered_sentences = self.get_clusters(papers_standardized, searched_papers)
        joblib.dump(clustered_sections, self.dump_dir + 'clustered_sections.dmp')
        joblib.dump(clustered_sentences, self.dump_dir + 'clustered_sentences.dmp')

        '''
        self.print_fn("clusters extracted")
        self.print_fn("clustered_sentences types:"+ str(np.unique([str(type(txt)) for k,txt in clustered_sentences.items()])))
        self.print_fn("clustered_sentences example:")
        self.print_fn(random.sample(list(clustered_sections.items()), 1)[0])
        self.print_fn("clustered_sections types:"+ str(np.unique([str(type(txt)) for k,txt in clustered_sections.items()])))
        self.print_fn("clustered_sections example:")
        self.print_fn(random.sample(list(clustered_sections.items()), 1)[0])
        '''
        clustered_sections['abstract'] = abstract_block
        clustered_sections['introduction'] = intro_block
        joblib.dump(clustered_sections, self.dump_dir + 'research_sections.dmp')

        self.print_fn("\n- Building conclusion.. ")
        conclusion_block = self.get_conclusion(clustered_sections)
        joblib.dump(conclusion_block, self.dump_dir + 'conclusion_block.dmp')
        clustered_sections['conclusion'] = conclusion_block
        '''
        self.print_fn("conclusion_block type:"+ str(type(conclusion_block)))
        self.print_fn("conclusion_block:")
        self.print_fn(conclusion_block)
        '''
        if query is None:
            query = self.generate_title(' '.join([v for v in clustered_sections.values()]))

        survey_file = 'A_Survey_on_' + query.replace(' ', '_') + '.txt'
        survey_file = Path(self.dump_dir).resolve() / survey_file
        self.build_doc(clustered_sections, papers_standardized, query=query, filename=str(survey_file))

        self.survey_print_fn("\n-citation-network: ")
        self.survey_print_fn(cites)

        shutil.copytree('arxiv_data/', self.dump_dir + '/arxiv_data/')
        assert (os.path.exists(survey_file))
        
        zip_name = 'arxiv_dumps_'+query.replace(' ', '_')+'.zip'
        zip_name = Path(self.dump_dir).parent.resolve() / zip_name
        self.zip_outputs(self.dump_dir, str(zip_name))
        self.print_fn("\n- Survey complete.. \nSurvey file path :" + str(survey_file) + 
                      "\nAll outputs zip path :" + str(zip_name))

        return str(zip_name.resolve()), str(zip_name.resolve())
