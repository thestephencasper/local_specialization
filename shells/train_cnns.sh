#!/usr/bin/env bash

for i in {1..5}
do

    echo "######## CNN: MNIST ########"
    python -m src.train_nn with cnn_config dataset_name=mnist extract_activations=True
#     echo "######## CNN: FASHION ########"
#     python -m src.train_nn with cnn_config dataset_name=fashion extract_activations=True

#     echo "######## CNN: MNIST+DROPOUT ########"
#     python -m src.train_nn with cnn_config dataset_name=mnist with_dropout=True
#     echo "######## CNN: FASHION+DROPOUT ########"
#     python -m src.train_nn with cnn_config dataset_name=fashion with_dropout=True

#     echo "######## CNN: MNIST+L1REG ########"
#     python -m src.train_nn with cnn_config dataset_name=mnist with_l1reg=True
#     echo "######## CNN: FASHION+L1REG ########"
#     python -m src.train_nn with cnn_config dataset_name=fashion with_l1reg=True
#
#     echo "######## CNN: MNIST+L2REG ########"
#     python -m src.train_nn with cnn_config dataset_name=mnist with_l2reg=True
#     echo "######## CNN: FASHION+L2REG ########"
#     python -m src.train_nn with cnn_config dataset_name=fashion with_l2reg=True
#
#     echo "######## CNN: CNN-CLUST-INIT-MNIST ########"
#     python -m src.train_nn with cnn_config dataset_name=mnist init_modules=10
#     echo "######## CNN: CNN-CLUST-INIT-FASHION ########"
#     python -m src.train_nn with cnn_config dataset_name=fashion init_modules=10
#
#     echo "######## CNN: STACKED-SAME-MNIST ########"
#     python -m src.train_nn with cnn_config dataset_name=stacked_same_mnist extract_activations=True
#     echo "######## CNN: STACKED-SAME-FASHION ########"
#     python -m src.train_nn with cnn_config dataset_name=stacked_same_fashion extract_activations=True
#
#     echo "######## CNN: STACKED-MNIST ########"
#     python -m src.train_nn with cnn_config dataset_name=stacked_mnist extract_activations=True
#     echo "######## CNN: STACKED-FASHION ########"
#     python -m src.train_nn with cnn_config dataset_name=stacked_fashion extract_activations=True

    echo "######## CNN: MNIST+LUCID ########"
    python -m src.train_nn with cnn_config dataset_name=mnist lucid=True extract_activations=True
#     echo "######## CNN: MNIST+DROPOUT+LUCID ########"
#     python -m src.train_nn with cnn_config dataset_name=mnist with_dropout=True lucid=True
#     echo "######## CNN: CLUST-INIT-MNIST+LUCID ########"
#     python -m src.train_nn with cnn_config dataset_name=mnist init_modules=10 lucid=True
#     echo "######## CNN: CLUST-INIT-MNIST + DROPOUT+LUCID ########"
#     python -m src.train_nn with cnn_config dataset_name=mnist init_modules=10 with_dropout=True lucid=True

done

# echo "######## CNN: MNIST+DROPOUT ########"
# python -m src.train_nn with cnn_config dataset_name=mnist with_dropout=True extract_activations=True
# echo "######## CNN: FASHION+DROPOUT ########"
# python -m src.train_nn with cnn_config dataset_name=fashion with_dropout=True extract_activations=True
#
# echo "######## CNN: STACKED-SAME-MNIST ########"
# python -m src.train_nn with cnn_config dataset_name=stacked_same_mnist extract_activations=True
# echo "######## CNN: STACKED-MNIST ########"
# python -m src.train_nn with cnn_config dataset_name=stacked_mnist extract_activations=True
# echo "######## CNN: STACKED-SAME-MNIST +DROPOUT ########"
# python -m src.train_nn with cnn_config dataset_name=stacked_same_mnist with_dropout=True extract_activations=True
# echo "######## CNN: STACKED-MNIST +DROPOUT########"
# python -m src.train_nn with cnn_config dataset_name=stacked_mnist with_dropout=True extract_activations=True
