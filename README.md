# CLDC 
 
CLDC is a task for reviewing crosslingual word embedding .
 


# Requirement
 
* Python 3.6.5


 
# Usage
```bash
mkdir CLDC/data
mkdir CLDC/saved_models
```
 
Please git clone ted-cldc corpora from [here](http://www.clg.ox.ac.uk/tedcldc.html)

 
```bash
mv ted-cldc CLDC/data/
```


```bash
./ted-cldc_set.sh
```
 
```bash
python3 main.py word_vec_path1 word_vec_path2
```

or if you want to use this system in jupyter lab, please look at CLDC/run.ipynb
 

 
# Author
 
* Sosuke
* Twitter : https://twitter.com/ponyo_ponyo115
 

 

 
Thank you!