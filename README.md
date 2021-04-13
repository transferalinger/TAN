# Transfer Alignment Network for Blind Unsupervised Domain Adaptation

This package provides implementations of Transfer Alignment Network.
Transfer Alignment Network is a stack of autoencoder, transfer aligner
layers and mlp networks.

## Code structure
`./src/model`: python scripts for model definition

`./src/train`: python scripts for train and test models defined in `./src/model`

`./src/demo`: demo shell script for batch execution of training codes in `./src/train`

## Naming convention
**auto_encoder** (ae): Autoencoder

**mlp**: Multilayer Perceptron

**v1**: Multilayer Perceptron on top of Autoencoder

**transfer aligner** (aligner): Transfer Alignment Layer connecting source and
target Autoencoder

**v2**: Multilayer Perceptron on top of Transfer Alignment Layer and Autoencoder

## Usage
source ae_train -> source v1_train -> target ae_train -> target mn_train -> target v2_test

## Dependencies

* Numpy
* TensorFlow

## Data description
* Following data files are in `data/sensorless`
    * `sensorless_source_unlabeled.csv`: unlabeled source data
    * `sensorless_source_labeled.csv`: labeled source data
    * `sensorless_source_test.csv`: source test data
    * `sensorless_target_unlabeled.csv`: unlabeled target data
    * `sensorless_target_test.csv`: target test data

## Output folder structure
* Output of the model training is stored in the directory specified in argument log_dir.
* Output folder sturcture
	* `train.log`: log file having results
	* `test.log`: log file having results for every test step
	* `best.log`: log file having only the best result
	* `hyperparameter`: json file having hyperparameter configuration for the current step
	* `weight/`: folder having csv files of trained weights
* log file columns
	* columns in log files for autoencoder and aligner training are loss, test loss, test_diff, test_rel_diff
	* columns in log files for classifier training are loss, test loss, test_accuracy, auc_roc, auc_pr

## Demo
* There is a demo script `src/demo/script.sh`
    * Input: `data/sensorless`
    * Output: `src/results/step1`, `src/results/step2`, `src/results/step3`, `src/results/step4`, `src/results/test`
        * log files for step1, 3, 4 are loss, test loss, test_diff, test_rel_diff
        * log files for step2 is loss, test loss, test_accuracy, auc_roc, auc_pr
        * log files for test is test loss, test_accuracy, auc_roc, auc_pr
