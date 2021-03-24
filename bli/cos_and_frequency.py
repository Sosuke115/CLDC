from collections import Counter

from tqdm import tqdm
from multiprocessing import Pool
import util

import numpy as np
import matplotlib.pyplot as plt

from get_cosines import GetCosie
from word_counter import WordCounter

import scipy.stats


class DrawScatter:
    def __init__(self, dict_path, trg_count, cosines_dict, pscosines_dict):
        self.eval_dict_pairs = [
            x.lower().split("\t") for x in util.load_lines(dict_path)
        ]
        self.trg_count = trg_count
        self.cosines_dict = cosines_dict
        self.pscosines_dict = pscosines_dict

    def _set_list(self):

        # 頻度数とcos類似度
        x_list, y_list = [], []

        for word_src, word_trg in self.eval_dict_pairs:
            if (
                word_trg in self.trg_count
                and self.pscosines_dict[(word_src, word_trg)] != None
                and self.cosines_dict[(word_src, word_trg)] != None
            ):
                x_list.append(self.trg_count[word_trg])
                y_list.append(
                    self.pscosines_dict[(word_src, word_trg)]
                    - self.cosines_dict[(word_src, word_trg)]
                )
        self.x_list = x_list
        self.y_list = y_list

    def draw(self):

        self._set_list()

        plt.scatter(
            self.x_list,
            self.y_list,
            c="black",
            alpha=0.5,
        )
        plt.xlim(0, 10000)
        plt.grid(True)

    def get_pearsonr(self):
        self._set_list()
        return scipy.stats.pearsonr(self.x_list, self.y_list)


def main():

    # TODO fix hard cording
    text_path = "/data/11/nishikawa/wiki_experiments/text_data/train.en"
    vec_path = "/data/11/nishikawa/wiki_experiments/embeddings/train.en.s0.vec"
    en_counter = WordCounter(text_path, vec_path)
    en_word_count = en_counter.get_word_count()

    dict_path = "/data/local/nishikawa/xling_eval/bli_datasets/en-fr/yacle.test.freq.2k.fr-en.tsv"
    trg_vec_path = (
        "/data/11/nishikawa/wiki_experiments/embeddings/train.en.50k.s0.vec.enfr.map"
    )
    src_vec_path = (
        "/data/11/nishikawa/wiki_experiments/embeddings/train.fr.50k.s0.vec.enfr.map"
    )

    en_fr_cosines = GetCosie(dict_path, src_vec_path, trg_vec_path)
    en_fr_cosines_dict = en_fr_cosines.get_cosines()

    dict_path = "/data/local/nishikawa/xling_eval/bli_datasets/en-fr/yacle.test.freq.2k.fr-en.tsv"
    trg_vec_path = "/data/11/nishikawa/wiki_experiments/pseudo_fren/train.enps.50k.true.s0.vec.enps-fr.map"
    src_vec_path = (
        "/data/11/nishikawa/wiki_experiments/embeddings/train.fr.50k.s0.vec.enps-fr.map"
    )

    enps_fr_cosines = GetCosie(dict_path, src_vec_path, trg_vec_path)
    enps_fr_cosines_dict = enps_fr_cosines.get_cosines()

    draw_graph_en_fr = DrawScatter(
        dict_path, en_word_count, en_fr_cosines_dict, enps_fr_cosines_dict
    )
    draw_graph_en_fr.draw()

    print(draw_graph_en_fr.get_pearsonr()[0])


if __name__ == "__main__":
    main()