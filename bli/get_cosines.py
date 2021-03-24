from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool
import util
import numpy as np
import matplotlib.pyplot as plt


class GetCosie:
    def __init__(self, dict_path, src_vec_path, trg_vec_path):
        self.dict_path = dict_path
        self.src_vec_path = src_vec_path
        self.trg_vec_path = trg_vec_path

    def _load_vectors(self):
        print("load vectors")
        vocab_src, embs_src = util.load_embs(self.src_vec_path)
        vocab_trg, embs_trg = util.load_embs(self.trg_vec_path)

        self.vocab_src = vocab_src
        self.vocab_trg = vocab_trg
        self.embs_src = embs_src
        self.embs_trg = embs_trg

    def _load_dict(self):
        print("load dict")
        self.eval_dict_pairs = [
            x.lower().split("\t") for x in util.load_lines(self.dict_path)
        ]

    def _get_cosine_sim_words(
        self, word_src, word_trg, vocab_src, vocab_trg, src_embs_norm, trg_embs_norm
    ):
        if word_src not in vocab_src and word_src.lower() not in vocab_src:
            return None
        if word_trg not in vocab_trg and word_trg.lower() not in vocab_trg:
            return None
        word_src_emb = src_embs_norm[
            vocab_src[word_src]
            if word_src in vocab_src
            else vocab_src[word_src.lower()]
        ]
        word_trg_emb = trg_embs_norm[
            vocab_trg[word_trg]
            if word_trg in vocab_trg
            else vocab_trg[word_trg.lower()]
        ]
        return cos_sim(word_src_emb, word_trg_emb)

    def get_cosines(self):

        self._load_vectors()
        self._load_dict()

        word_to_cosine_dict = dict()
        print("calc cosine")
        for word_src, word_trg in self.eval_dict_pairs:
            sim = self._get_cosine_sim_words(
                word_src,
                word_trg,
                self.vocab_src,
                self.vocab_trg,
                self.embs_src,
                self.embs_trg,
            )
            word_to_cosine_dict[(word_src, word_trg)] = sim
        return word_to_cosine_dict
