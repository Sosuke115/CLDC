# CLDC (Cross-Lingual Document Crassification)
 
CLDC is a task for evaluating crosslingual word embedding .
 


# Requirement
 
* Python 3.6.5
* Pytorch 1.1.0

# Test Dataset
[ted-cldc](http://www.clg.ox.ac.uk/tedcldc.html)

[amazon](https://github.com/facebookresearch/MLDoc)

[MLdoc](https://webis.de/data/webis-cls-10.html)


 
# Usage
```bash
mkdir CLDC/data
mkdir CLDC/saved_models
```
 



 
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
