# Introduction


This directory contains the implementation of downstream tasks (Cross-Lingual Document Crassification) for bilingual word embeddings in our paper (**Data Augmentation with Unsupervised Machine Translation Improves the Structural Similarity of Cross-lingual Word Embeddings**).

https://aclanthology.org/2021.acl-srw.17.pdf

 


# Requirement
 
* Python 3.6.5
* Pytorch 1.1.0

# Dataset

[amazon](https://github.com/facebookresearch/MLDoc)

[MLdoc](https://webis.de/data/webis-cls-10.html)

[ted-cldc](http://www.clg.ox.ac.uk/tedcldc.html)




# Prepare Data

```bash
mkdir CLDC/saved_models
```

##### mldoc, amazon

```bash
unzip CLDC/data/mldoc.zip
unzip CLDC/data/amazon.zip
```

##### ted
```bash
mv ted-cldc CLDC/data/
```

```bash
./ted-cldc_set.sh
```
 
 
# Usage


 
```bash
python3 main.py src_word_vec_path1 trg_word_vec_path2 --lang1 en --lang2 fr --dataset mldoc --model_name test --word_dim 512 
```

or if you want to use this system in jupyter lab, please look at CLDC/run.ipynb




 
# Author
 
* Sosuke
* website : https://sosuke115.github.io
* Twitter : https://twitter.com/ponyo_ponyo115

# Reference

```bibtex
@inproceedings{nishikawa-etal-2021-data,
    title = "Data Augmentation with Unsupervised Machine Translation Improves the Structural Similarity of Cross-lingual Word Embeddings",
    author = "Nishikawa, Sosuke  and
      Ri, Ryokan  and
      Tsuruoka, Yoshimasa",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing: Student Research Workshop",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-srw.17",
    doi = "10.18653/v1/2021.acl-srw.17",
    pages = "163--173",
}
```
 

 

 
Thank you!
