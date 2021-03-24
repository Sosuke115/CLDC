from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool
import util
import numpy as np
import matplotlib.pyplot as plt


# vec_file、text_fileを読み込んで単語数辞書を返すクラス
class WordCounter:
    def __init__(self, text_file_path, vec_file_path):
        self.text_file_path = text_file_path
        self.vec_file_path = vec_file_path

    def get_words_list(self) -> (list, int):

        file_obj = vec = open(self.vec_file_path, "r", errors="replace")
        words_list = []
        line = file_obj.readline()
        cnt = 0
        num = 0
        while line:
            list_c = line.strip().split()
            if cnt == 0:
                num = list_c[0]
            if cnt != 0:
                words_list.append(list_c[0])
            line = file_obj.readline()
            cnt += 1
        file_obj.close()

        return words_list, num

    def get_text_file(self):

        with open(self.text_file_path) as f:
            l_strip = [s.strip() for s in tqdm(f.readlines())]
        return l_strip

    def get_word_count(self):

        print("loading text_list")

        text_list = self.get_text_file()

        print("loading word_list")

        word_list, num = self.get_words_list()

        print("count word")

        counter = Counter()
        for sentence in tqdm(text_list):
            counter.update(sentence.split())

        new_counter = dict()
        for word in word_list:
            new_counter[word] = counter[word]
        return new_counter