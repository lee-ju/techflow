# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import gensim
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import copy

class nlp_preps:
    """
    Read more in the 'github.com/lee-ju/techflow#techflownlp_tech'
    
    Parameters
    ----------
    1. `x`: Texts to be DTM.
    """
    def __init__(self, x):
        self.x = x

    def dtmx(self, max_features=100, use_ptrain=True, use_weight=True, ptrain_path=None):
        """
        Read more in the 'github.com/lee-ju/techflow#techflownlp_tech'
    
        Parameters
        ----------
        1. `max_features`: If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus. (default: 100)
        2. `use_ptrain`: True if using the pre-trained word embedding model, False if using only tf-idf. (default: True)
        3. `use_weight`: True if embedding values are used as weights, False otherwise. Select only when use_ptrain is True. (default: True)
        4. `ptrain_path`: Path for pre-trained word embedding model. (default: None)
        """
        self.max_features = int(max_features)
        self.use_ptrain = use_ptrain
        self.use_weight = use_weight
        self.ptrain_path = ptrain_path
        self.vocab = []
        self.out_x = []
        
        stop_words = stopwords.words('english')
        vectorizer = TfidfVectorizer(max_features=self.max_features,
                                     stop_words=stop_words)
        dtm = vectorizer.fit_transform(self.x)
        dtm_arr = dtm.toarray()

        N = dtm_arr.shape[0]
        V = dtm_arr.shape[1]
        
        if self.use_ptrain:
            w2v_model = gensim.models.KeyedVectors.load_word2vec_format(self.ptrain_path, binary=True)

            vocab_eng = list(w2v_model.index_to_key)
            
            for i in tqdm(range(len(vocab_eng))):
                vocabulary = str(vocab_eng[i])
                vocab.append(vocabulary)

            W = w2v_model.vectors
            term = vectorizer.get_feature_names()[:]
            
            for i in tqdm(range(0, len(term))):
                Term_i = term[i]
                try:
                    v_idx = vocab.index(Term_i)
                    find_vec = W[v_idx]
                except ValueError:
                    find_vec = np.zeros((W.shape[1]))
                self.out_x.append(find_vec)
            out_arr = np.array(self.out_x)
            if use_weight:
                emb_x = np.matmul(dtm_arr, out_arr)
            else:
                for i in range(N):
                    for j in range(V):
                        if dtm_arr[i][j] == 0:
                            dtm_arr[i][j] = 0
                        else:
                            dtm_arr[i][j] = 1    
                emb_x = np.matmul(dtm_arr, out_arr)

        else:
            emb_x = copy.deepcopy(dtm_arr)
            
        return emb_x
