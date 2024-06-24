# PubMed-MultiLabel-Text-Classification

## Solution

The underlying objective is to use the dataset's `abstractText` column to create a multi-label text classification model. One-hot encoded columns serve as a representation of the target labels. Because of the nature of the challenge, a transformer-based model, BERT in particular, is selected due to its robustness in problems involving natural language processing.

In the `notebooks/analysis.ipynb` an initial analysis was perfomed not only to figure out the necessary data cleaning and pre-processing parts, but also to identify possible limitations of the dataset, such as the imbalancement either of the single labels or the joined one-hot labels, i.e. sequences.

The initial idea is to use the encoder part of the transformers and have a simple Linear layer at the end to serve as the classifier. BERT was selected, because of the smaller size, ease of use and the fact that it serves as a good baseline for this project. Additionally, the `cased` version was selected, in order to preserve case-sensitive information. A lot of times in medical papers, and abstracts, proper nouns and technical terms are used.

BERT tokenizer was used, since it is designed to match the tokenization used during BERT's pretraining phase. A notable advantage of this tokenizer is that it utilizes special tokens that help BERT understand the structure and boundaries of the input text. The ideal `max_length` was found based on the 95th percentile, in order to cover most of the data and avoid excessive truncation. However, a smaller one was chosen, due to computation limitations.

It is left as future work to experiment with TF-IDF encoding and a simple clasifier such as SVM or generate more synthetic training data to mitigate imbalancement.

## Setup

```
git clone https://github.com/nikarav/PubMed-MultiLabel-Text-Classification.git
```

Then, using conda create a new environment with python version `3.11.9`. For example:

```
conda create -n pubmed python=3.11.9
```

Afterwards, activate the environment and install the requirements. If you want to only test the fastapi then you can just install the `requirements_infer.txt`. In order to install all the requirements, then perform the following:

```
conda activate pubmed
pip install -r requirements_dev.txt
```

> **_NOTE:_** Include the project in the python path to avoid any relative import errors. This can be done such as:

```
export PYTHONPATH=/path/to/this/project:$PYTHONPATH
```

In the future the project could be packaged to avoid altering the python path.

### Git-LFS (Optional)

In the folder `weights` the weights of the best performing model across the epochs are saved. However, due to the big size of the file, `git-lfs` was used. Therefore, to downlaod the weights, first you need to install [git-lfs](https://git-lfs.com/). Afterwards, navigate in the root directory of this project and type:

```
git lfs install
git lfs fetch fetch --all
git lfs pull
git pull
```

## Dataset Preparation

The dataset used in this project can be found [here](https://huggingface.co/datasets/owaiskha9654/PubMed_MultiLabel_Text_Classification_Dataset_MeSH).

Before the training and/or validation, the dataset needs to be prepared. In the folder `scripts`, the file `prepare_data.py` aims to download the dataset and prepare the training, validation, and testing splits. Specifically, duplicate entries are removed based on the abstractText column, labels are axtracted from `meshRoot`, converting them into one-hot encoded format.

Afterwards, the dataset is shuffled to ensure randomness and split into training, validation, and test sets using `MultilabelStratifiedShuffleSplit` to handle class imbalance better than basic stratify method of `sklearn`. This library can be found at [iterative-stratification](https://github.com/trent-b/iterative-stratification). Finally, the separate binary label columns are combined into a list for easier data handling during training and validation. You can achieve this by running:

```
python scripts/prepare_data.py --output /path/to/saveFolder --val_pct 0.1 --tst_pct 0.05 -v
```

The `saveFolder` needs be created beforehand in this case and the script will automatically fetch the dataset from the web. Alternatively, if it already downloaded you can alter the `--data_path` argument and specify the path to the dataset. The flags `val_pct` and `tst_pct` specify the percentage of the validation/ test split and `-v` increases the verbose level of print pessages to debug.

<!-- [An Internal Link](/guides/content/editing-an-existing-page) -->

## Training

The train script is located in `src/train.py`. First, the `config/config_train.yaml` file needs to be configured by specifying the paths of the data splits as well as parameters such as `batch`, `epochs`, `max_length` etc.

An example usage for running the train script is the following:

```
python src/train.py --config config/config_train.yaml --cuda -v
```

During training, the weights are saved(the path can be changed in the config file), and validation is performed after each training epoch. The validation metrics used are macro Precision, macro Recall and macro f1 for a threshold of **0.5**.

## Validation

The validation script is in `src/val.py` and the config file is `config/config_val.yaml`. It performs validation by loading the specified weights. The metrics are the same used in the training script and the results are printed after computation is completed. An example of utilizing the val script is:

```
python src/val.py ----config config/config_val.yaml --cuda -v
```

**It should be noted that throughout this project specific seed has been set for reproducible data splits and train/val results.**

## Results

Below are the results during the training of the model. It was trained for 5 epochs and the best results were obtained at the last epoch.
| Epoch | Train Loss | Val Loss | Val Precision (macro) | Val Recall (macro) | Val F1 (macro) |
|-------|-------------|----------|-----------------------|--------------------|----------------|
| 5 | 0.25491944931627036 | 0.2749195954738519 | 0.8023743033409119 | 0.6891066431999207 | 0.7215514779090881 |

In the folder `weights/` a file named `best.pth` and it contains the weights of the best performing model.

## FastAPI

The inference script is exposed via a RESTful API using `fastapi`. Specifically, the code is located in the `app` directory and in order to start the service type the following:

```
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

By default, the app will init the model using `weights/best.pth`. In case you wish to try different weights, then set up the environment variable `MODEL_PATH` with the path to the model weigths.

The functionality can be tested by sending a request to `/predict` endpoint, either by editing line 9 of the file `test/test_api.py` and executing it or by using `curl` to send a POST request to the endpoint with the JSON data included, such as:

```
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"texts":"Sample text for classification."}'
```

## Future Work

- Try TF-IDF encoding and other zero-shot/few-shot models
- Enrich repo with pre-commit hooks
- Include a scheduler to training and integrate with a mlops logging service.
- Package the project.
