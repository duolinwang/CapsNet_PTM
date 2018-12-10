# CapsNet_PTM

CapsNet for Protein Post-translational Modification site prediction. It is implemented by deep learning library Keras2.1.1 and Tensorflow backend
# Training and testing data
The 10-fold cross-validation training and tesing data for each PTM used in the paper are in folder all_PTM_raw_data.
Each subfolder contains 10-fold annotated training sequences (metazoa_sequence_annotated_training_X.fasta, sequences that have more than 30% sequence identity with the testing set were removed) and corresponding annotated testing sequences (metazoa_sequences_cross_testing_annotated_X.fasta) and testing sequences without annotation (metazoa_sequences_cross_testing_X.fasta)

# Installation

  - Download codes by 
  ```sh
  git clone https://github.com/duolinwang/CapsNet_PTM
  ```
  - Installation has been tested in Linux and Mac OS X with Python 2.7. 
  - Since the package is written in python 2.7, [python 2.7](https://www.python.org/downloads/ ) with the pip tool must be installed first. 
It uses the following dependencies:
numpy,  scipy, pandas, h5py, keras version=2.1.1
You can install these packages first, by the following commands:

```sh
pip install pandas
pip install numpy
pip install scipy
pip install h5py
pip install -v keras==2.1.1
pip install tensorflow (or GPU supported tensorflow, refer to https://www.tensorflow.org/install/ for instructions)
```
 - This is the Tensorflow version, you must change the backend to TensorFlow.
If you have run Keras at least once, you will find the Keras configuration file at:
$HOME/.keras/keras.json
If it isn’t there, you can create it. 
Change the default configuration file into:
```sh
{	
    "image_dim_ordering": "th",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```
# Running on GPU or CPU

>If you want to use GPU, you also need to install [CUDA]( https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn); refer to their websites for instructions. 
CPU is only suitable for prediction not training. 
#### For custom training:
```sh
python train_models.py -input [custom training data in fasta format] -output-prefix [prefix of pre-trained model] -residue-types [custom specified residue types]
```
For details of other parameters, run:
```sh
python train_models.py --help
```
or
```sh
python train_models.py -h
```

####Example

python train_models.py -input train_example_data.fasta -output-prefix example_models -residue-types S,T

#### Custom prediction from custom general models and custom kinase-specific models:
```sh
python predict.py -input [custom predicting data in fasta format] -model-prefix [prefix of pre-trained model] -output [custom specified file for predicting results] 
```

####Example

python predict.py -input test_example_data.fasta -model-prefix example_models -output test_example_result

### Citation：
Please cite the following paper for using MusiteDeep:
Duolin Wang, Yanchun Liang, Dong Xu*, Capsule Network for Protein Post-translational Modification Site Prediction.

License
----
GNU v2.0
