# Neural Models in NLP

This project consists of two components. In the first component, we debug through a near-functional implementation of a neural model. Four errors are corrected in the code and the model can deal with a sentence level classification task with an accuracy of 84.4\% after five epoch training.

In the second component, we use seq-2-seq framework to deal with three tasks, copying, reversing and sorting data sequences. Several questions are answered in this part. Then, we add two commonly seen attention method -- multiplicative and addictive to deal with the same tasks and analyze its asymptotic performance.
## Getting Started

Our project is implemented by Python 3.7

### Prerequisites

Since we use `PyTorch` to train the neural network models, you need to make sure the library has been installed.
If not, you can use pip to install the library.
```
pip3 install torch torchvision
```
We also use `tqdm` to show the progress during the training. 
```
pip3 install tqdm
```

## Project Structure

- **data**: Training data and test data provided by TA.
- **model**: In order to save time, we store the trained models into this directory if you open a `SAVEMODEL` macro in the main.py.
- **Project3_Resources**: Requirements.
- **test**: Save logs and test metrics by varying different parameters.
- **toy_data**: Toy datasets including copy, reverse, and sort. For each dataset we generate a training set, validation set and test set.
- **Codes**:
    - accuracy.py: Python script showing accuracy based on the test logs.
    - broken_main.py: main function that needs to be debugged
    - broken_model.py: model class that needs to be debugged
    - main.py: experiments & attention 
    - model.py: neural networks model
    - generate_toy_data.py & generate_vocab.py: python scripts that will be used by toy.sh
    - toy.sh: shell script that generates random toy datasets by inputing different parameters.
- README.md

## Running the tests

Explain how to run the automated tests for this system

### Generate Toy DataSet
The toy dataset is randomly generated based on different parameters.
```
./toy.sh DATA_TYPE NUMEXAMPLES VOCAB_SIZE MAX_LEN SPECIAL_LEN
```
The parameters you can vary is:
1. DATA_TYPE: type of dataset which includes copy, reverse and sort.
2. NUMEXAMPLES: number of sequences in the dataset.
3. VOCAB_SIZE: normal dictionary size.
4. MAX_LEN: the maximum of each sequence length.
5. SPECIAL_LEN: special dictionary size.

### Neural Networks with Attention
You are able to run the networks with or without attention by command line.

- Neural networks without attention
```
python main.py
```
- Neural networks with additive attention
```
python main.py add
```
- Neural networks with multiplicative attention
```
python main.py mul
```

## Authors

* **Jialu Li** - *Debug & Documents* - jl3855@cornell.edu
* **Charlie Wang** - *Attention & Documents* - qw248@cornell.edu
* **Ziyun Wei** - *Experiments & Documents* - zw555@cornell.edu


## License

No license

