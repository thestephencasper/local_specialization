# <NOTEBOOK_PATH> <OUTPUT_PATH>
PAPERMILL := papermill --cwd ./notebooks
# NBX := jupyter nbconvert --ExecutePreprocessor.timeout=-1 --execute --to notebook --inplace

.PHONY: clean-datasets clean-models clean-all mkdir-results models test all
.PHONY: mlp-clustering mlp-lesion mlp-double-lesion make mlp-plots

clean-datasets:
	rm -rf datasets

clean-models:
	rm -rf training_runs_dir models

clean-all: clean-datasets clean-models

mkdir-results:
	mkdir -p results

# Running `pytest src` causes to the weird `sacred` and `tensorflow` import error:
# ImportError: Error while finding loader for 'tensorflow' (<class 'ValueError'>: tensorflow.__spec__ is None)
# https://github.com/IDSIA/sacred/issues/493
test:
	pytest src/tests/test_lesion.py
	pytest src/tests/test_utils.py
	pytest src/tests/test_cnn.py
	pytest src/tests/test_spectral_clustering.py

all: datasets models test mlp-analysis

mlp-clustering: mkdir-results
	$(PAPERMILL) ./notebooks/mlp-clustering.ipynb ./notebooks/mlp-clustering.ipynb

mlp-clustering-stability: mkdir-results
	$(PAPERMILL) ./notebooks/mlp-clustering-stability.ipynb ./notebooks/mlp-clustering-stability.ipynb

mlp-clustering-stability-n-clusters: mkdir-results
	$(PAPERMILL) ./notebooks/mlp-clustering-stability-different-n_clusters.ipynb ./notebooks/mlp-clustering-stability-different-n_clusters-K=2.ipynb -p N_CLUSTERS 2
	$(PAPERMILL) ./notebooks/mlp-clustering-stability-different-n_clusters.ipynb ./notebooks/mlp-clustering-stability-different-n_clusters-K=7.ipynb -p N_CLUSTERS 7
	$(PAPERMILL) ./notebooks/mlp-clustering-stability-different-n_clusters.ipynb ./notebooks/mlp-clustering-stability-different-n_clusters-K=10.ipynb -p N_CLUSTERS 10
    
mlp-learning-curve: mkdir-results
	$(PAPERMILL) ./notebooks/mlp-learning-curve.ipynb ./notebooks/mlp-learning-curve.ipynb

# Using 10 clusters
mlp-lesion-TEN: mkdir-results
	$(PAPERMILL) ./notebooks/mlp-lesion-test-TEN.ipynb ./notebooks/mlp-lesion-test-TEN.ipynb

mlp-double-lesion: mkdir-results
	# $(PAPERMILL) ./notebooks/mlp-double-lesion-test.ipynb ./notebooks/mlp-double-lesion-test-MNIST.ipynb -p MODEL_TAG MNIST
	# $(PAPERMILL) ./notebooks/mlp-double-lesion-test.ipynb ./notebooks/mlp-double-lesion-test-MNIST+DROPOUT.ipynb -p MODEL_TAG MNIST+DROPOUT
	# $(PAPERMILL) ./notebooks/mlp-double-lesion-test.ipynb ./notebooks/mlp-double-lesion-test-FASHION.ipynb -p MODEL_TAG FASHION
	# $(PAPERMILL) ./notebooks/mlp-double-lesion-test.ipynb ./notebooks/mlp-double-lesion-test-FASHION+DROPOUT.ipynb -p MODEL_TAG FASHION+DROPOUT

mlp-plots:
	$(PAPERMILL) ./notebooks/mlp-plots.ipynb ./notebooks/mlp-plots.ipynb

mlp-analysis: mlp-clustering mlp-clustering-stability mlp-clustering-stability-n-clusters mlp-learning-curve mlp-lesion mlp-double-lesion mlp-plots

cnn-clustering: mkdir-results
	$(PAPERMILL) ./notebooks/cnn-clustering.ipynb ./notebooks/cnn-clustering.ipynb

lucid-make-dataset-mlp: mkdir-results
	$(PAPERMILL) ./notebooks/lucid_make_dataset_mlp.ipynb ./notebooks/lucid_make_dataset_mlp.ipynb
lucid-make-dataset-cnn: mkdir-results
	$(PAPERMILL) ./notebooks/lucid_make_dataset_cnn.ipynb ./notebooks/lucid_make_dataset_cnn.ipynb
lucid-make-dataset-cnn-vgg: mkdir-results
	$(PAPERMILL) ./notebooks/lucid_make_dataset_cnn_vgg.ipynb ./notebooks/lucid_make_dataset_cnn_vgg.ipynb
lucid-make-dataset-cnn-vgg-k8: mkdir-results
	$(PAPERMILL) ./notebooks/lucid_make_dataset_cnn_vgg_k8.ipynb ./notebooks/lucid_make_dataset_cnn_vgg_k8.ipynb
lucid-make-dataset-cnn-vgg-k12: mkdir-results
	$(PAPERMILL) ./notebooks/lucid_make_dataset_cnn_vgg_k12.ipynb ./notebooks/lucid_make_dataset_cnn_vgg_k12.ipynb
lucid-prep-imagenet: mkdir-results
	$(PAPERMILL) ./notebooks/lucid_prep_imagenet.ipynb ./notebooks/lucid_prep_imagenet.ipynb
lucid-make-dataset-imagenet: mkdir-results
	$(PAPERMILL) ./notebooks/lucid_make_dataset_imagenet.ipynb ./notebooks/lucid_make_dataset_imagenet.ipynb
lucid-results-all: mkdir-results
	$(PAPERMILL) ./notebooks/lucid_results_all.ipynb ./notebooks/lucid_results_all.ipynb
lucid-results-altk: mkdir-results
	$(PAPERMILL) ./notebooks/lucid_results_altk.ipynb ./notebooks/lucid_results_altk.ipynb

deleteme: mkdir-results
	$(PAPERMILL) ./notebooks/deleteme.ipynb ./notebooks/deleteme.ipynb

lesion-test-mlp: mkdir-results
	$(PAPERMILL) ./notebooks/lesion_test_mlp.ipynb ./notebooks/lesion_test_mlp.ipynb
lesion-test-cnn: mkdir-results
	$(PAPERMILL) ./notebooks/lesion_test_cnn.ipynb ./notebooks/lesion_test_cnn.ipynb
lesion-test-cnn-vgg: mkdir-results
	$(PAPERMILL) ./notebooks/lesion_test_cnn_vgg.ipynb ./notebooks/lesion_test_cnn_vgg.ipynb
lesion-test-cnn-vgg-unreg: mkdir-results
	$(PAPERMILL) ./notebooks/lesion_test_cnn_vgg_unreg.ipynb ./notebooks/lesion_test_cnn_vgg_unreg.ipynb
lesion-test-imagenet: mkdir-results
	$(PAPERMILL) ./notebooks/lesion_test_imagenet.ipynb ./notebooks/lesion_test_imagenet.ipynb
lesion-results-all: mkdir-results
	$(PAPERMILL) ./notebooks/lesion_results_all.ipynb ./notebooks/lesion_results_all.ipynb
lesion-results-altk: mkdir-results
	$(PAPERMILL) ./notebooks/lesion_results_altk.ipynb ./notebooks/lesion_results_altk.ipynb

deep-dive: mkdir-results
	$(PAPERMILL) ./notebooks/deep_dive.ipynb ./notebooks/deep_dive.ipynb

###
lesion-test-mlp-altk: mkdir-results
	$(PAPERMILL) ./notebooks/lesion_test_mlp_altk.ipynb ./notebooks/lesion_test_mlp_altk.ipynb
lesion-test-cnn-altk: mkdir-results
	$(PAPERMILL) ./notebooks/lesion_test_cnn_altk.ipynb ./notebooks/lesion_test_cnn_altk.ipynb
lesion-test-cnn-vgg-altk: mkdir-results
	$(PAPERMILL) ./notebooks/lesion_test_cnn_vgg_altk.ipynb ./notebooks/lesion_test_cnn_vgg_altk.ipynb
lesion-test-cnn-vgg-k8: mkdir-results
	$(PAPERMILL) ./notebooks/tmp_lesion_test_cnn_vgg_k8.ipynb ./notebooks/tmp_lesion_test_cnn_vgg_k8.ipynb
lesion-test-cnn-vgg-k12: mkdir-results
	$(PAPERMILL) ./notebooks/tmp_lesion_test_cnn_vgg_k12.ipynb ./notebooks/tmp_lesion_test_cnn_vgg_k12.ipynb
lesion-test-imagenet-altk: mkdir-results
	$(PAPERMILL) ./notebooks/lesion_test_imagenet_altk.ipynb ./notebooks/lesion_test_imagenet_altk.ipynb
lesion-test-taxonomy: mkdir-results
	$(PAPERMILL) ./notebooks/lesion_test_taxonomy.ipynb ./notebooks/lesion_test_taxonomy.ipynb
lucid-deleteme: mkdir-results
	$(PAPERMILL) ./notebooks/deleteme_after_nov24.ipynb ./notebooks/deleteme_after_nov24.ipynb
###

layerwise-clusters: mkdir-results
	$(PAPERMILL) ./notebooks/layerwise_clusters.ipynb ./notebooks/layerwise_clusters.ipynb

top-ims-all: mkdir-results
	$(PAPERMILL) ./notebooks/top_ims_all.ipynb ./notebooks/top_ims_all.ipynb

act-cluster-imagenet: mkdir-results
	$(PAPERMILL) ./notebooks/cluster_imagenet_activations.ipynb ./notebooks/cluster_imagenet_activations.ipynb

clustering-factors-mlp: mkdir-results
	$(PAPERMILL) ./notebooks/clustering_factors_mlp.ipynb ./notebooks/clustering_factors_mlp.ipynb

clustering-factors-clust-grad: mkdir-results
	$(PAPERMILL) ./notebooks/clustering_factors_clust_grad.ipynb ./notebooks/clustering_factors_clust_grad.ipynb

clustering-factors-cnn: mkdir-results
	$(PAPERMILL) ./notebooks/clustering_factors_cnn.ipynb ./notebooks/clustering_factors_cnn.ipynb

clustering-factors-cnn-vgg: mkdir-results
	$(PAPERMILL) ./notebooks/clustering_factors_cnn_vgg.ipynb ./notebooks/clustering_factors_cnn_vgg.ipynb

clustering-factors-imagenet: mkdir-results
	$(PAPERMILL) ./notebooks/clustering_factors_imagenet.ipynb ./notebooks/clustering_factors_imagenet.ipynb

plotting-main: mkdir-results
	$(PAPERMILL) ./notebooks/plotting_main.ipynb ./notebooks/plotting_main.ipynb

random-init-ncuts: mkdir-results
	$(PAPERMILL) ./notebooks/random_init_ncuts.ipynb ./notebooks/random_init_ncuts.ipynb
