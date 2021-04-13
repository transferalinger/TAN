#!/bin/bash
cd ../train

# dataset information
DATA_DIR_NAME="sensorless"
DATA_NAME="sensorless"
INPUT=48;
LABEL_INDEX=0;
DATA_INDEX=`seq -s " " 1 48`;
OUT=11;

# directory setting
DATA_PREFIX="../../data/$DATA_DIR_NAME"
LOG_DIR_PREFIX="../results/"
SOURCE_TEST="$DATA_PREFIX/$DATA_NAME"_source_test.csv
TARGET_TEST="$DATA_PREFIX/$DATA_NAME"_target_test.csv

# model architecture
MLP_HIDDEN=("300 300 300 300 300");
MN_HIDDEN=("100 50 100");
AE_HIDDEN=("80 40 20");
AE_Z=10;

# Step1: train autoencoder on source unlabeled data
LOG_DIR1=$LOG_DIR_PREFIX"/step1"
SOURCE_UNLABEL="$DATA_PREFIX/$DATA_NAME"_source_unlabeled.csv
SOURCE_UNLABEL_N=`wc -l $SOURCE_UNLABEL | awk '{k+=$1}END{print k};'`;

ARGS1="--batch 100 \
--lr 0.001 \
--optimizer adam \
--epoch 2 \
--reg_param 0 \
--input $INPUT \
--z $AE_Z \
--hidden $AE_HIDDEN \
--train $SOURCE_UNLABEL \
--train_n $(($SOURCE_UNLABEL_N)) \
--test $SOURCE_TEST \
--data_index $DATA_INDEX \
--log_dir $LOG_DIR1 \
--display_step 200 \
--early_stop 100 \
--device gpu
"
python ae_train.py $ARGS1;

# Step2: train mlp on top of autoencoder on source labeled data
LOG_DIR2=$LOG_DIR_PREFIX"/step2"
SOURCE_LABEL="$DATA_PREFIX/$DATA_NAME"_source_labeled.csv
SOURCE_LABEL_N=`wc -l $SOURCE_LABEL | awk '{k+=$1}END{print k};'`;

ARGS2="--batch 100 \
--lr 0.001 \
--epoch 20 \
--input $INPUT \
--out $OUT \
--hidden $MLP_HIDDEN \
--train $SOURCE_LABEL \
--train_n $(($SOURCE_LABEL_N)) \
--test $SOURCE_TEST \
--data_index $DATA_INDEX \
--label_index $LABEL_INDEX \
--log_dir $LOG_DIR2 \
--test_step 200 \
--early_stop 100 \
--encoder_path $LOG_DIR1 \
--encoder_lr 0.001
"
python v1_train.py $ARGS2;

# Step3: fine-tune autoencoder on target unlabeled data
LOG_DIR3=$LOG_DIR_PREFIX"/step3"
TARGET_UNLABEL="$DATA_PREFIX/$DATA_NAME"_target_unlabeled.csv
TARGET_UNLABEL_N=`wc -l $TARGET_UNLABEL | awk '{k+=$1}END{print k};'`;

ARGS3="--batch 100 \
--lr 0.001 \
--epoch 20 \
--reg_param 0 \
--input $INPUT \
--z $AE_Z \
--hidden $AE_HIDDEN \
--train $TARGET_UNLABEL \
--train_n $(($TARGET_UNLABEL_N)) \
--test $TARGET_TEST \
--data_index $DATA_INDEX \
--log_dir $LOG_DIR3 \
--display_step 200 \
--early_stop 100 \
--use_pretrained \
--pretrain_path $LOG_DIR2 \
--device gpu
"
python ae_train.py $ARGS3;

# Step4: train aligner on target unlabeled data
LOG_DIR4=$LOG_DIR_PREFIX"/step4"

ARGS4="--batch 100 \
--lr 0.001 \
--epoch 20 \
--reg_param 0 \
--reg_layer_idx 2 \
--hidden $MN_HIDDEN \
--train $TARGET_UNLABEL \
--train_n $(($TARGET_UNLABEL_N)) \
--test $TARGET_TEST \
--data_index $DATA_INDEX \
--label_index $LABEL_INDEX \
--log_dir $LOG_DIR4 \
--test_step 200 \
--early_stop 100 \
--encoder1_path $LOG_DIR2 \
--encoder2_path $LOG_DIR3 \
--encoder_lr 0.001
"
python aligner_train.py $ARGS4;

# Step5: test mlp on top of autoencoder on target labeled data
LOG_DIR5=$LOG_DIR_PREFIX"/test"
TARGET_LABEL="$DATA_PREFIX/$DATA_NAME"_target_test.csv
TARGET_LABEL_N=`wc -l $TARGET_LABEL | awk '{k+=$1}END{print k};'`;

ARGS5="--batch 100 \
--lr 0.001 \
--epoch 1 \
--input $INPUT \
--out $OUT \
--hidden $MLP_HIDDEN \
--train $TARGET_LABEL \
--train_n $(($TARGET_LABEL_N)) \
--test $TARGET_TEST \
--data_index $DATA_INDEX \
--label_index $LABEL_INDEX \
--log_dir $LOG_DIR5 \
--test_step 200 \
--early_stop 100 \
--encoder_path $LOG_DIR4 \
--matchnet_path $LOG_DIR4 \
--encoder_lr 0.001 \
--use_pretrained \
--pretrain_path $LOG_DIR2
"
python v2_test.py $ARGS5;
