#!/usr/bin/env bash

# datasets (also be sure to follow instructions in the README for imagenet)
bash shells/make_datasets.sh

# train
bash shells/train_mlps.sh
bash shells/train_cnns.sh
bash shells/train_cnn_vggs.sh

# featue vis with lucid
make lucid-make-dataset-mlp
make lucid-make-dataset-cnn
make lucid-make-dataset-cnn-vgg
make lucid-prep-imagenet
make lucid-make-dataset-imagenet
make lucid-results-all

make lucid-make-dataset-cnn-vgg-k8
make lucid-make-dataset-cnn-vgg-k12
make lucid-results-altk

# lesion tests
make lesion-test-mlp
make lesion-test-cnn
make lesion-test-cnn-vgg
make lesion-test-imagenet
make lesion-results-all

make lesion-test-cnn-vgg-altk
make lesion-results-altk

