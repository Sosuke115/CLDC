# CLDC 
 
CLDC is a task for reviewing crosslingual word embedding .
 


# Requirement
 
* Python 3.6.5
* Pytorch 1.1.0


 
# Usage
```bash
mkdir CLDC/data
mkdir CLDC/saved_models
```
 
Please git clone ted-cldc corpora from [here](http://www.clg.ox.ac.uk/tedcldc.html)

Please git clone amazon corpora from [here](https://github.com/facebookresearch/MLDoc)

Please git clone MLdoc corpora from [here](https://webis.de/data/webis-cls-10.html)




 
```bash
mv ted-cldc CLDC/data/
```

```bash
./ted-cldc_set.sh
```
 
```bash
python3 main.py word_vec_path1 word_vec_path2 --lang1 en --lang2 fr --dataset ted --model_name test --word_dim 512 
```

or if you want to use this system in jupyter lab, please look at CLDC/run.ipynb
 

 
# Author
 
* Sosuke
* Twitter : https://twitter.com/ponyo_ponyo115
 

 

 
Thank you!