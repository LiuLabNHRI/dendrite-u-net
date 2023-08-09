This is code to train the U-Net based model for dendrite segmentation used in
the paper **A photo-switchable assay system for dendrite degeneration and repair
in Drosophila melanogaster**.


## Environment setup
Require [Anaconda](https://www.anaconda.com/products/individual).

1. Clone the repository and cd into the folder.
2. Create a conda environment by `conda env create -f environment.yml`.
3. Activate the environment by `conda activate tf2`.


## Usage
The main function to train and apply a model for segmentation is `src/segmentation.py`.

~~~bash
python src/segmentation.py -h
~~~

This will show the usage of this function:

~~~bash
usage: segmentation.py [-h] [--mode MODE] [--gpu_id GPU]
                       [--src_model_yaml SRC_MODEL_YAML]
                       [--test_yaml TEST_YAML]
                       task_yaml

positional arguments:
  task_yaml             yaml file of the task

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE           TRAIN, TRANSFER, EVAL or TEST
  --gpu_id GPU          ID of GPU to use
  --src_model_yaml SRC_MODEL_YAML
                        Yaml of source model
  --test_yaml TEST_YAML
                        Yaml of test data
~~~


### Add the provided pre-trained model
1. Download and unzip the provided model folder (model_062519).
2. Move the model folder into the `models` subfolder:

~~~bash
.
├── configs
├── data
├── models
│   └── larva_062519
│       ├── checkpoint
│       ├── weights-0475.ckpt.data-00000-of-00002
│       ├── weights-0475.ckpt.data-00001-of-00002
│       └── weights-0475.ckpt.index
├── src
├── Quantify_morphology.ipynb
├── environment.yml
└── README.md
~~~


### Train a new model
After creating a model configuration (e.g. model_061921.yaml) in the `configs` folder, execute:

~~~bash
python src/segmentation.py model_061921.yaml --mode TRAIN
~~~

The model weights at different epochs will be saved in the `models` folder.


### Evalue the trained model
Provide the model configuration (e.g. larva_062519.yaml), and change mode to "EVAL".

~~~bash
python src/segmentation.py larva_062519.yaml --mode EVAL
~~~

This will create predicted segmentation in the `results` folder.


### Apply the trained model to predict segmentation
The data being predicted need to be specified in `test_yaml` and stored in the `configs` folder (e.g. fig_s1.yaml).

~~~bash
python src/segmentation.py larva_062519.yaml --mode TEST --test_yaml fig_s1.yaml
~~~

The predicted segmentation maps will be saved in the `results` folder.


### Quantify morphology
We provide a jupyter notebook (`Quantify_morphology.ipynb`) to post-process segmentation maps and quantify total dendrite length and tip numbers. After creating segmentation maps in the `results` folder, open and execute `Quantify_morphology.ipynb`.
