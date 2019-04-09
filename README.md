# ML-prototypes

## Overview

A temporary repo for data processing and tensorflow model training. The important/successful code will eventually be moved to the webserver repo (https://github.com/ASLive/webserver) for ad-hoc prediction of client input.

## Dependencies

* Tensorflow version 1.12  
* Virtualenv is recommended - https://docs.python-guide.org/dev/virtualenvs/  
* Python version 3.6  
* To install requirements, run `pip3 install -r requirements.txt`   
* Might need install python3-tk: `sudo apt-get install python3-tk`

## Setup
 
* download and extract hand3d training data: 
   * https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html
   * exact link: https://lmb.informatik.uni-freiburg.de/data/RenderedHandpose/RHD_v1-1.zip
   * extract `RHD_published_v2/` to parent folder of repo
* run hand3d setup `python3 setup_hand3d.py`
* test hand3d was successful `python3 test_hand3d.py`
* download and extract asl letter training data:
   * https://drive.google.com/file/d/1_UR6pySf_IaMzhqCyC7OG8ubkAGI99OM/view?usp=sharing
   * extract `training_data/` to parent folder of repo
* download and extract asl training data processed with hand3d
   * alternatively run `setup_asl.py` which could take a long while
   * https://drive.google.com/file/d/1box3Pp8PPLZ0WeUOy0igsFSXSpSHU7DD/view?usp=sharing
   * extract all files to `<this_repo>/pickle/*`
* hand3d training data and asl training data locations can be changed in `settings.py`

## Usage
```
   python3 main.py # load model files if available, else retrain and save
   python3 main.py retrain # retrain model and save
   python3 main.py retrain no-save # retrain the model but don't save model
```

## Pizza Limerick

```
A pizza is wonderful food.
To not eat a slice would be rude.
You must be polite.
So eat every bite,
And make sure it is properly chewed.
```
