#!/usr/bin/env bash

# train networks without extracting activations
for i in {1..5}
do

    echo "######## MLP: MNIST ########"
    python -m src.train_nn with mlp_config dataset_name=mnist extract_activations=True
#     echo "######## MLP: FASHION ########"
#     python -m src.train_nn with mlp_config dataset_name=fashion extract_activations=True

#     echo "######## MLP: MNIST + DROPOUT ########"
#     python -m src.train_nn with mlp_config dataset_name=mnist with_dropout=True
#     echo "######## MLP: FASHION + DROPOUT ########"
#     python -m src.train_nn with mlp_config dataset_name=fashion with_dropout=True
#
#     echo "######## MLP: MNIST + L1REG ########"
#     python -m src.train_nn with mlp_config dataset_name=mnist with_l1reg=True
#     echo "######## MLP: FASHION + L1REG ########"
#     python -m src.train_nn with mlp_config dataset_name=fashion with_l1reg=True
#
#     echo "######## MLP: MNIST + L2REG ########"
#     python -m src.train_nn with mlp_config dataset_name=mnist with_l2reg=True
#     echo "######## MLP: FASHION + L2REG ########"
#     python -m src.train_nn with mlp_config dataset_name=fashion with_l2reg=True
#
#     echo "######## MLP: CLUST-INIT-MNIST ########"
#     python -m src.train_nn with mlp_config dataset_name=mnist init_modules=10
#     echo "######## MLP: CLUST-INIT-FASHION ########"
#     python -m src.train_nn with mlp_config dataset_name=fashion init_modules=10
#
#     echo "######## MLP: HALVES-SAME-MNIST ########"
#     python -m src.train_nn with mlp_config dataset_name=halves_same_mnist extract_activations=True
#     echo "######## MLP: HALVES-SAME-FASHION ########"
#     python -m src.train_nn with mlp_config dataset_name=halves_same_fashion extract_activations=True
#
#     echo "######## MLP: HALVES-MNIST ########"
#     python -m src.train_nn with mlp_config dataset_name=halves_mnist extract_activations=True
#     echo "######## MLP: HALVES-FASHION ########"
#     python -m src.train_nn with mlp_config dataset_name=halves_fashion extract_activations=True

    echo "######## MLP: MNIST+LUCID ########"
    python -m src.train_nn with mlp_config dataset_name=mnist epochs=30 lucid=True extract_activations=True
#     echo "######## MLP: MNIST + DROPOUT + LUCID ########"
#     python -m src.train_nn with mlp_config dataset_name=mnist epochs=30 with_dropout=True lucid=True
#     echo "######## MLP: CLUST-INIT-MNIST + LUCID ########"
#     python -m src.train_nn with mlp_config dataset_name=mnist init_modules=10 epochs=30 lucid=True
#     echo "######## MLP: CLUST-INIT-MNIST + DROPOUT + LUCID ########"
#     python -m src.train_nn with mlp_config dataset_name=mnist init_modules=10 with_dropout=True \
#     epochs=30 lucid=True
#
#     echo "######## MLP: POLY ########"
#     python -m src.train_nn with mlp_regression_config dataset_name=poly
#     echo "######## MLP: POLY + L1REG########"
#     python -m src.train_nn with mlp_regression_config dataset_name=poly with_l1reg=True
#     echo "######## MLP: POLY + L2REG########"
#     python -m src.train_nn with mlp_regression_config dataset_name=poly with_l2reg=True

done

# echo "######## MLP: MNIST DROPOUT ########"
# python -m src.train_nn with mlp_config dataset_name=mnist with_dropout=True extract_activations=True
# echo "######## MLP: FASHION DROPOUT ########"
# python -m src.train_nn with mlp_config dataset_name=fashion with_dropout=True extract_activations=True
#
# echo "######## MLP: HALVES-SAME-MNIST ########"
# python -m src.train_nn with mlp_config dataset_name=halves_same_mnist extract_activations=True
# echo "######## MLP: HALVES-MNIST ########"
# python -m src.train_nn with mlp_config dataset_name=halves_mnist extract_activations=True
# echo "######## MLP: HALVES-SAME-MNIST+DROPOUT ########"
# python -m src.train_nn with mlp_config dataset_name=halves_same_mnist with_dropout=True extract_activations=True
# echo "######## MLP: HALVES-MNIST+DROPOUT ########"
# python -m src.train_nn with mlp_config dataset_name=halves_mnist with_dropout=True extract_activations=True
