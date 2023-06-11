## Code Implementation based on paper :
Image Processing for Detecting Botnet Attacks: A Novel Approach for Flexibility and Scalability

https://ieeexplore.ieee.org/document/9945055


dataset:

https://www.kaggle.com/datasets/mkashifn/nbaiot-dataset?resource=download

### requirements before run
- install [Anaconda](https://www.anaconda.com/)
- install [tensorflow](https://www.tensorflow.org/install/pip)
- download [the dataset](https://www.kaggle.com/datasets/mkashifn/nbaiot-dataset?resource=download)
- activate conda virtual env for this project 
```
conda activate {vritual env}
```
- install the required packages
```
pip install -r requirements.txt
```
- make required directories
```
mkdir generated_image result saved_model
```
- change "BASE_DATASET_DIR" on config.py to the directory of dataset in your computer

### convert image v1
example script for convert dataset into image using v1: 
```
python run_convert_v1.py -f mean -d 1.benign.csv
```
- -f , filter query for dataset columns (optional)
- -d, dataset filename include extension (required)

### convert image v2
example script for convert dataset into image using v2:
```
python run_convert_v2.py -f variance -t 0.15
```
- -f, filter query for dataset columns (optional)
- -t, threshold for correlation, above threshold are the selected values 

### training process
Training the AI model using generated image.
#### using v1 generated image
example script training using v1 generated image:
```
python run_train.py -p v1 -f variance
```
- -p, for choosing generator program type
- -f, filter query for dataset columns (optional)
#### using v2 generated image
example script training using v2 generated image:
```
python run_train.py -p v2 -f weight -t 0.15
```
- -p, for choosing generator program type
- -f, filter query for dataset columns (optional)
- -t, threshold for correlation, above threshold are the selected values

### output directory structure
#### Generated image
```
generated_image
├── v1
│   ├── all
│   └── {filter query 1}
│   └── (filter query 2)
│   └── -----
├── v2
│   ├── all
│   └── {filter query 1}
│       └── {threshold 1}
│       └── {threshold 2}
│       └── {threshold 3}
│       └── ----------
│   └── (filter query 2)
│       └── {threshold 1}
│       └── {threshold 2}
│       └── {threshold 3}
│       └── ----------
│   └── -----
```
#### Result
```
result
├── v1_{filter query 1}.csv
├── v1_{filter query 2}.csv
├── v1_{filter query 3}.csv
├── ---------
├── v2_{filter query 1}_{threshold 1}.csv
├── v2_{filter query 1}_{threshold 2}.csv
├── v2_{filter query 2}_{threshold 1}.csv
├── v2_{filter query 3}_{threshold 2}.csv
├── ---------
```
#### Saved model
```
saved_model
├── v1_{filter query 1}
├── v1_{filter query 2}
├── v1_{filter query 3}
├── ---------
├── v2_{filter query 1}_{threshold 1}
├── v2_{filter query 1}_{threshold 2}
├── v2_{filter query 2}_{threshold 1}
├── v2_{filter query 3}_{threshold 2}
├── ---------
```

### additional notes:
- change "BENIGN_FILENAME" on config.py if you want to use another file in dataset folder for benign data
- change "MIRAI_FILENAME"on config.py if you want to use another file in dataset folder for mirai data
- using pearson correlation for v2 program