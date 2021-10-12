import pickle
from tempfile import mkstemp

import lucid.modelzoo.vision_models as models
import lucid.optvis.objectives as objectives
import lucid.optvis.render as render
import numpy as np
import tensorflow as tf
from classification_models.keras import Classifiers
from decorator import decorator
from lucid.misc.io import show
from lucid.modelzoo.vision_base import Model
from lucid.optvis import param
from lucid.optvis.objectives_util import _make_arg_str, _T_force_NHWC
from scipy.stats import entropy

from src.cnn import CNN_MODEL_PARAMS, CNN_VGG_MODEL_PARAMS
from src.experiment_tagging import get_model_path
from src.lesion.experimentation import load_model2
from src.spectral_cluster_model import run_clustering_imagenet, get_dense_sizes
from src.utils import splitter, compute_pvalue, get_model_paths, fisher_stat, combine_ps, chi2_categorical_test
from src.activations import get_labels_imagenet_activations

MIN_SIZE = 2
MAX_PROP = 0.99

IMAGE_SIZE = 28
IMAGE_SIZE_CIFAR10 = 32
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
IMAGE_SHAPE_CIFAR10 = (IMAGE_SIZE_CIFAR10, IMAGE_SIZE_CIFAR10, 3)


def lucid_model_factory(pb_model_path=None,
                        model_image_shape=IMAGE_SHAPE,
                        model_input_name='dense_input',
                        model_output_name='dense_4/Softmax',
                        model_image_value_range=(0, 1)):
    """Build Lucid model object."""

    if pb_model_path is None:
        _, pb_model_path = mkstemp(suffix='.pb')

    # Model.suggest_save_args()

    # Save tf.keras model in pb format
    # https://www.tensorflow.org/guide/saved_model
    Model.save(
        pb_model_path,
        image_shape=model_image_shape,
        input_name=model_input_name,
        output_names=[model_output_name],
        image_value_range=model_image_value_range)

    class MyLucidModel(Model):
        model_path = pb_model_path
        # labels_path = './lucid/mnist.txt'
        # synsets_path = 'gs://modelzoo/labels/ImageNet_standard_synsets.txt'
        # dataset = 'ImageNet'
        image_shape = model_image_shape
        # is_BGR = True
        image_value_range = model_image_value_range
        input_name = model_input_name

    lucid_model = MyLucidModel()
    lucid_model.load_graphdef()

    return lucid_model


def print_model_nodes(lucid_model):
    graph_def = tf.GraphDef()
    with open(lucid_model.model_path, 'rb') as f:
        graph_def.ParseFromString(f.read())
        for node in graph_def.node:
            print(node.name)


class CustomObjective(objectives.Objective):

    def obj_abs(self):

        objective_func = lambda T: tf.math.abs(self(T))
        name = f"abs_{self.name}"
        description = f"Abs({self.description})"
        return CustomObjective(objective_func, name=name, description=description)


def custom_wrap_objective(require_format=None, handle_batch=False):
    @decorator
    def inner(f, *args, **kwds):
        objective_func = f(*args, **kwds)
        objective_name = f.__name__
        args_str = " [" + ", ".join([_make_arg_str(arg) for arg in args]) + "]"
        description = objective_name.title() + args_str
        def process_T(T):
            if require_format == "NHWC":
                T = _T_force_NHWC(T)
            return T
        return CustomObjective(lambda T: objective_func(process_T(T)),
                               objective_name, description)
    return inner


@custom_wrap_objective(require_format='NHWC')
def custom_channel(layer, n_channel, batch=None):

    @objectives.handle_batch(batch)
    def inner(T):
        return tf.reduce_mean(T(layer)[..., n_channel])

    return inner


def render_vis_with_loss(model, objective_f, size, optimizer=None,
                         transforms=[], thresholds=(64,), print_objectives=None,
                         relu_gradient_override=True):

    param_f = param.image(size)
    images = []
    losses = []

    with param_f.graph.as_default() as graph, tf.Session() as sess:

        T = render.make_vis_T(model, objective_f, param_f=param_f, optimizer=optimizer,
                              transforms=transforms, relu_gradient_override=relu_gradient_override)
        print_objective_func = render.make_print_objective_func(print_objectives, T)
        loss, vis_op, t_image = T("loss"), T("vis_op"), T("input")
        tf.global_variables_initializer().run()

        for i in range(max(thresholds)+1):
            loss_, _ = sess.run([loss, vis_op])
            if i in thresholds:
                vis = t_image.eval()
                images.append(vis)
                losses.append(loss_)
                # if display:
                #     print(f'loss: {loss_}')
                #     print_objective_func(sess)
                #     show(vis)

    tf.compat.v1.reset_default_graph()

    return images[-1], losses[-1]


def make_lucid_dataset(model_tag, lucid_net, all_labels, is_unpruned, local=False, transforms=[],
                       n_random=19, min_size=MIN_SIZE, max_prop=MAX_PROP, display=False,
                       savedir='/project/nn_clustering/datasets/', savetag=''):

    if 'cnn' in model_tag.lower():
        cnn_params = CNN_VGG_MODEL_PARAMS if 'vgg' in str(model_tag).lower() else CNN_MODEL_PARAMS
        layer_sizes = [cl['filters'] for cl in cnn_params['conv']]
        # layer_names = ['conv2d/Relu'] + [f'conv2d_{i}/Relu' for i in range(1, len(layer_sizes))]
        layer_names = ['conv2d/BiasAdd'] + [f'conv2d_{i}/BiasAdd' for i in range(1, len(layer_sizes))]
    else:  # it's an mlp
        layer_sizes = [256, 256, 256, 256]
        # layer_names = ['dense/Relu'] + [f'dense_{i}/Relu' for i in range(1, len(layer_sizes))]
        layer_names = ['dense/BiasAdd'] + [f'dense_{i}/BiasAdd' for i in range(1, len(layer_sizes))]
    if not is_unpruned:
        layer_names = ['prune_low_magnitude_' + ln for ln in layer_names]

    assert sum(layer_sizes) == len(all_labels), f'sum(layer_sizes)={sum(layer_sizes)}, but len(all_labels)={len(all_labels)}'

    labels_in_layers = [np.array(lyr_labels) for lyr_labels in list(splitter(all_labels, layer_sizes))]

    max_images = []  # to be filled with images that maximize cluster activations
    max_idxs = []  # to be filled with indices for true sub-cluster neurons
    random_max_images = []  # to be filled with images that maximize random units activations
    random_max_idxs = []  # to be filled with indices for random sub-cluster neurons
    max_losses = []  # to be filled with losses
    random_max_losses = []  # to be filled with losses
    sm_sizes = []  # list of submodule sizes
    sm_layer_sizes = []
    sm_layers = []  # list of layer names
    sm_clusters = []  # list of clusters

    imsize = IMAGE_SIZE_CIFAR10 if 'vgg' in model_tag.lower() else IMAGE_SIZE

    for layer_name, labels, layer_size in zip(layer_names, labels_in_layers, layer_sizes):
        max_size = max_prop * layer_size
        for clust_i in range(int(max(all_labels)+1)):
            sm_binary = labels == clust_i
            sm_size = sum(sm_binary)
            if sm_size <= min_size or sm_size >= max_size:  # skip if too big or small
                continue

            sm_sizes.append(sm_size)
            sm_layer_sizes.append(layer_size)
            sm_layers.append(layer_name)
            sm_clusters.append(clust_i)

            # print(f'{model_tag}, layer: {layer_name}')
            # print(f'submodule_size: {sm_size}, layer_size: {layer_size}')

            sm_idxs = [i for i in range(layer_size) if sm_binary[i]]
            max_idxs.append(np.array(sm_idxs))
            max_obj = sum([custom_channel(layer_name, unit).obj_abs() for unit in sm_idxs])

            max_im, max_loss = render_vis_with_loss(lucid_net, max_obj, size=imsize, transforms=transforms)
            max_images.append(max_im)
            max_losses.append(max_loss)
            if display:
                print(f'loss: {round(max_loss, 3)}')
                show(max_im)

            rdm_losses = []
            rdm_ims = []
            for _ in range(n_random):  # random max results
                rdm_idxs = np.random.choice(np.array(range(layer_size)), size=sm_size, replace=False)
                random_max_obj = sum([custom_channel(layer_name, unit).obj_abs() for unit in rdm_idxs])
                random_max_im, random_max_loss = render_vis_with_loss(lucid_net, random_max_obj,
                                                                      size=imsize, transforms=transforms)
                random_max_images.append(random_max_im)
                random_max_losses.append(random_max_loss)
                random_max_idxs.append(rdm_idxs)
                rdm_ims.append(np.squeeze(random_max_im))
                rdm_losses.append(round(random_max_loss, 3))
            if display:
                print(f'random losses: {rdm_losses}')
                show(np.hstack(rdm_ims))

    max_images = np.squeeze(np.array(max_images))
    random_max_images = np.squeeze(np.array(random_max_images))
    max_losses = np.array(max_losses)
    random_max_losses = np.array(random_max_losses)

    results = {'max_images': max_images,
               'random_max_images': random_max_images,
               'max_losses': max_losses,
               'random_max_losses': random_max_losses,
               'sm_sizes': sm_sizes, 'sm_layer_sizes': sm_layer_sizes,
               'sm_layers': sm_layers, 'sm_clusters': sm_clusters,
               'max_idxs': max_idxs, 'random_max_idxs': random_max_idxs}

    if is_unpruned:
        suff = '_unpruned_max_data'
        suff2 = '_unpruned_acts'
    else:
        suff = '_pruned_max_data'
        suff2 = '_pruned_acts'

    with open(savedir + model_tag + suff + savetag + '.pkl', 'wb') as f:
        pickle.dump(results, f)
    with open(savedir + model_tag + suff2 + savetag + '.pkl', 'wb') as f:
        pickle.dump(all_labels, f)


def evaluate_visualizations(model_tag, tag_sfx, rep, is_unpruned, suff2='',
                            min_size=MIN_SIZE, max_prop=MAX_PROP,
                            data_dir='/project/nn_clustering/datasets/', savetag=''):

    if is_unpruned:
        suff = f'{rep}_unpruned_max_data{suff2}.pkl'
        suff2 = f'{rep}_unpruned_acts{suff2}.pkl'
    else:
        suff = f'{rep}_pruned_max_data{suff2}.pkl'
        suff2 = f'{rep}_pruned_acts{suff2}.pkl'

    with open(data_dir + model_tag + tag_sfx + suff, 'rb') as f:
        data = pickle.load(f)
    with open(data_dir + model_tag + tag_sfx + suff2, 'rb') as f:
        all_labels = pickle.load(f)

    # unpack data
    max_images = data['max_images']
    if len(max_images.shape) == 3:
        max_images = np.expand_dims(max_images, 0)
    max_idxs = data['max_idxs']
    random_max_images = data['random_max_images']
    random_max_idxs = data['random_max_idxs']
    max_losses = data['max_losses']
    random_max_losses = data['random_max_losses']
    sm_sizes = data['sm_sizes']
    sm_layers = data['sm_layers']
    sm_layer_sizes = data['sm_layer_sizes']
    sm_clusters = data['sm_clusters']
    n_examples = len(sm_sizes)
    n_max_min = int(len(max_images) / n_examples)
    n_random = int(len(random_max_images) / n_examples)
    input_side = max_images.shape[1]

    # flatten all inputs if mlp
    if 'mlp' in model_tag.lower():
        max_images = np.reshape(max_images, [-1, IMAGE_SIZE**2])
        random_max_images = np.reshape(random_max_images, [-1, IMAGE_SIZE**2])

    # get model
    model_dir = get_model_path(model_tag, filter_='all')[rep]
    model_path = get_model_paths(model_dir)[is_unpruned]
    model = load_model2(model_path)

    # get predictions
    max_preds = model.predict(max_images)
    random_max_preds = np.reshape(model.predict(random_max_images), (n_examples, n_random, -1))

    # get entropies
    max_entropies = np.array([entropy(pred) for pred in max_preds])
    random_max_entropies = np.array([[entropy(pred) for pred in reps] for reps in random_max_preds])

    # get dispersions
    max_dispersions = np.zeros_like(max_entropies)
    max_covs = np.zeros_like(max_entropies)
    random_max_dispersions = np.zeros_like(random_max_entropies)

    if 'cnn' in model_tag.lower():
        cnn_params = CNN_VGG_MODEL_PARAMS if 'vgg' in str(model_tag).lower() else CNN_MODEL_PARAMS
        layer_sizes = [cl['filters'] for cl in cnn_params['conv']]
        layer_names = ['conv2d'] + [f'conv2d_{i}' for i in range(1, len(layer_sizes))]
    else:  # it's an mlp
        layer_sizes = [256, 256, 256, 256]
        layer_names = ['dense'] + [f'dense_{i}' for i in range(1, len(layer_sizes))]
    if not is_unpruned:
        layer_names = ['prune_low_magnitude_' + ln for ln in layer_names]
    labels_in_layers = [np.array(lyr_labels) for lyr_labels in list(splitter(all_labels, layer_sizes))]

    example_i = 0
    for layer_name, labels, layer_size in zip(layer_names, labels_in_layers, layer_sizes):
        max_size = max_prop * layer_size
        for clust_i in range(int(max(all_labels)+1)):
            sm_binary = labels == clust_i
            sm_size = sum(sm_binary)
            if sm_size <= min_size or sm_size >= max_size:  # skip if too big or small
                continue

            inp = model.input  # input placeholder
            outputs = [layer.output for layer in model.layers if layer._name==layer_name]
            functor = tf.keras.backend.function([inp, tf.keras.backend.learning_phase()], outputs)
            activations = np.squeeze(functor([np.expand_dims(max_images[example_i], 0), 0]))
            if 'cnn' in model_tag.lower():
                activations = np.array([np.linalg.norm(activations[:, :, i], 1)
                                        for i in range(activations.shape[2])])
            activations[activations<0] = 0
            true_std = np.std(activations[max_idxs[example_i]])
            true_cov = true_std / np.mean(activations[max_idxs[example_i]])
            all_random_stds = []
            for rdm_i in range(n_random):
                rdm_activations = np.squeeze(
                    functor([np.expand_dims(random_max_images[n_random*example_i+rdm_i], 0), 0]))
                if 'cnn' in model_tag.lower():
                    rdm_activations = np.array([np.linalg.norm(rdm_activations[:, :, i], 1)
                                                for i in range(rdm_activations.shape[2])])
                rdm_activations[rdm_activations < 0] = 0  # relu
                all_random_stds.append(np.std(rdm_activations[random_max_idxs[n_random*example_i+rdm_i]]))

            max_dispersions[example_i] = true_std
            max_covs[example_i] = true_cov
            random_max_dispersions[example_i] = np.array(all_random_stds)
            example_i += 1

    # get cov quartiles
    cov_quartiles = np.nanquantile(max_covs, [0.25, 0.5, 0.75])

    # reshape losses
    random_max_losses = np.reshape(random_max_losses, (n_examples, n_random))

    # get percentiles
    max_percentiles_entropy = np.array([compute_pvalue(max_entropies[i], random_max_entropies[i])
                                        for i in range(len(max_entropies))])
    max_percentiles_loss = np.array([compute_pvalue(max_losses[i], random_max_losses[i], side='right')
                                     for i in range(len(max_losses))])

    max_percentiles_dispersion = np.array([compute_pvalue(max_dispersions[i], random_max_dispersions[i])
                                           for i in range(len(max_dispersions))])

    # get effect sizes
    effect_factors_entropies = np.array([np.mean(max_entropies[i]) /
                                         (np.mean(max_entropies[i])+random_max_entropies[i])
                                         for i in range(len(max_entropies)) if max_entropies[i] > 0])
    effect_factor_entropy = np.nan_to_num(effect_factors_entropies)
    effect_factors_losses = np.array([np.mean(max_losses[i]) /
                                      (np.mean(max_losses[i])+random_max_losses[i])
                                      for i in range(len(max_losses)) if max_losses[i] > 0])
    effect_factor_loss = np.nan_to_num(effect_factors_losses)
    effect_factors_dispersions = np.array([np.mean(max_dispersions[i]) /
                                           (np.mean(max_dispersions[i])+random_max_dispersions[i])
                                         for i in range(len(max_dispersions)) if max_dispersions[i] > 0])
    effect_factor_dispersion = np.nan_to_num(effect_factors_dispersions)

    # get pvalues
    max_chi2_p_entropy = chi2_categorical_test(max_percentiles_entropy, n_random)
    max_fisher_p_entropy = combine_ps(max_percentiles_entropy, n_random)
    fisher_stat_entropy = fisher_stat(max_percentiles_entropy, n_random)
    max_chi2_p_loss = chi2_categorical_test(max_percentiles_loss, n_random)
    max_fisher_p_loss = combine_ps(max_percentiles_loss, n_random)
    fisher_stat_loss = fisher_stat(max_percentiles_loss, n_random)
    max_chi2_p_dispersion = chi2_categorical_test(max_percentiles_dispersion, n_random)
    max_fisher_p_dispersion = combine_ps(max_percentiles_dispersion, n_random)
    fisher_stat_dispersion = fisher_stat(max_percentiles_dispersion, n_random)

    results = {'percentiles': (max_percentiles_entropy, max_percentiles_loss, max_percentiles_dispersion),
               'effect_factors': (effect_factor_entropy, effect_factor_loss, effect_factor_dispersion),
               'chi2_ps': (max_chi2_p_entropy, max_chi2_p_loss, max_chi2_p_dispersion),
               'fisher_ps': (max_fisher_p_entropy, max_fisher_p_loss, max_fisher_p_dispersion),
               'fisher_stats': (fisher_stat_entropy, fisher_stat_loss, fisher_stat_dispersion),
               'cov_quartiles': cov_quartiles, 'sm_layers': sm_layers, 'sm_sizes': sm_sizes,
               'sm_layer_sizes': sm_layer_sizes, 'sm_clusters': sm_clusters}

    return results


###################################################################################
# ImageNet
###################################################################################

IMAGE_SIZE_IMAGENET = 224
VIS_NETS = ['vgg16', 'vgg19', 'resnet50']
VGG16_LAYER_MAP = {'block1_conv1': 'conv1_1/conv1_1', 'block1_conv2': 'conv1_2/conv1_2',
                   'block2_conv1': 'conv2_1/conv2_1', 'block2_conv2': 'conv2_2/conv2_2',
                   'block3_conv1': 'conv3_1/conv3_1', 'block3_conv2': 'conv3_2/conv3_2',
                   'block3_conv3': 'conv3_3/conv3_3', 'block4_conv1': 'conv4_1/conv4_1',
                   'block4_conv2': 'conv4_2/conv4_2', 'block4_conv3': 'conv4_3/conv4_3',
                   'block5_conv1': 'conv5_1/conv5_1', 'block5_conv2': 'conv5_2/conv5_2',
                   'block5_conv3': 'conv5_3/conv5_3'}
VGG19_LAYER_MAP = {'block1_conv1': 'conv1_1/conv1_1', 'block1_conv2': 'conv1_2/conv1_2',
                   'block2_conv1': 'conv2_1/conv2_1', 'block2_conv2': 'conv2_2/conv2_2',
                   'block3_conv1': 'conv3_1/conv3_1', 'block3_conv2': 'conv3_2/conv3_2',
                   'block3_conv3': 'conv3_3/conv3_3', 'block3_conv4': 'conv3_4/conv3_4',
                   'block4_conv1': 'conv4_1/conv4_1', 'block4_conv2': 'conv4_2/conv4_2',
                   'block4_conv3': 'conv4_3/conv4_3', 'block4_conv4': 'conv4_4/conv4_4',
                   'block5_conv1': 'conv5_1/conv5_1', 'block5_conv2': 'conv5_2/conv5_2',
                   'block5_conv3': 'conv5_3/conv5_3', 'block5_conv4': 'conv5_4/conv5_4'}
RESNET50_LAYER_MAP = {'conv0': 'resnet_v1_50/conv1/Relu',
                      'stage1_unit1_sc': 'resnet_v1_50/block1/unit_1/bottleneck_v1/Relu',
                      'stage1_unit2_conv3': 'resnet_v1_50/block1/unit_2/bottleneck_v1/Relu',
                      'stage1_unit3_conv3': 'resnet_v1_50/block1/unit_3/bottleneck_v1/Relu',
                      'stage2_unit1_sc': 'resnet_v1_50/block2/unit_1/bottleneck_v1/Relu',
                      'stage2_unit2_conv3': 'resnet_v1_50/block2/unit_2/bottleneck_v1/Relu',
                      'stage2_unit3_conv3': 'resnet_v1_50/block2/unit_3/bottleneck_v1/Relu',
                      'stage2_unit4_conv3': 'resnet_v1_50/block2/unit_4/bottleneck_v1/Relu',
                      'stage3_unit1_sc': 'resnet_v1_50/block3/unit_1/bottleneck_v1/Relu',
                      'stage3_unit2_conv3': 'resnet_v1_50/block3/unit_2/bottleneck_v1/Relu',
                      'stage3_unit3_conv3': 'resnet_v1_50/block3/unit_3/bottleneck_v1/Relu',
                      'stage3_unit4_conv3': 'resnet_v1_50/block3/unit_4/bottleneck_v1/Relu',
                      'stage3_unit5_conv3': 'resnet_v1_50/block3/unit_5/bottleneck_v1/Relu',
                      'stage3_unit6_conv3': 'resnet_v1_50/block3/unit_6/bottleneck_v1/Relu',
                      'stage4_unit1_sc': 'resnet_v1_50/block4/unit_1/bottleneck_v1/Relu',
                      'stage4_unit2_conv3': 'resnet_v1_50/block4/unit_2/bottleneck_v1/Relu',
                      'stage4_unit3_conv3': 'resnet_v1_50/block4/unit_3/bottleneck_v1/Relu'}
NETWORK_LAYER_MAP = {'vgg16': VGG16_LAYER_MAP, 'vgg19': VGG19_LAYER_MAP, 'resnet50': RESNET50_LAYER_MAP}


def get_clustering_info_imagenet(model_tag, num_clusters,
                                 savedir='/project/nn_clustering/results/',
                                 act_data_dir='/project/nn_clustering/datasets/imagenet2012'):

    assert model_tag in VIS_NETS

    clustering_results = run_clustering_imagenet(model_tag, num_clusters=num_clusters,
                                                 with_shuffle=False, eigen_solver='arpack')
    clustering_results_local = run_clustering_imagenet(model_tag, num_clusters=num_clusters, with_shuffle=False,
                                                       eigen_solver='arpack', local=True)

    layer_names = clustering_results['layer_names']
    conv_connections = clustering_results['conv_connections']
    layer_sizes = [cc[0]['weights'].shape[0] for cc in conv_connections[1:]]
    dense_sizes = get_dense_sizes(conv_connections)
    layer_sizes.extend(list(dense_sizes.values()))
    labels = clustering_results['labels']
    labels_local = clustering_results_local['labels']

    get_labels_imagenet_activations(model_tag, num_clusters, local=False)
    get_labels_imagenet_activations(model_tag, num_clusters, local=True)
    with open(act_data_dir + f'/{model_tag}_activations_local=False_k={num_clusters}.pkl', 'rb') as f:
        act_labels = pickle.load(f)
    with open(act_data_dir + f'/{model_tag}_activations_local=True_k={num_clusters}.pkl', 'rb') as f:
        act_labels_local = pickle.load(f)

    labels_in_layers = list(splitter(labels, layer_sizes))
    labels_in_layers_local = list(splitter(labels_local, layer_sizes))
    act_labels_in_layers = list(splitter(act_labels, layer_sizes))
    act_labels_in_layers_local = list(splitter(act_labels_local, layer_sizes))

    for nm, ly in zip(layer_names, layer_sizes):
        print(ly, nm)

    clustering_info = {'layers': layer_names,
                       'labels': labels_in_layers,
                       'labels_local': labels_in_layers_local,
                       'labels_act': act_labels_in_layers,
                       'labels_act_local': act_labels_in_layers_local}

    with open(savedir + model_tag + '_clustering_info.pkl', 'wb') as f:
        pickle.dump(clustering_info, f)


def make_lucid_imagenet_dataset(model_tag, use_activations, local=False, n_random=19,
                                min_size=MIN_SIZE, max_prop=MAX_PROP, display=False,
                                infodir='/project/nn_clustering/results/',
                                savedir='/project/nn_clustering/datasets/'):

    assert model_tag in VIS_NETS

    with open(infodir + model_tag + '_clustering_info.pkl', 'rb') as f:
        clustering_info = pickle.load(f)

    layer_names = clustering_info['layers']

    lil_key = 'labels'
    lil_key = lil_key + '_act' if use_activations else lil_key
    lil_key = lil_key + '_local' if local else lil_key
    labels_in_layers = [np.array(lyr_labels) for lyr_labels in clustering_info[lil_key]]

    layer_sizes = [len(labels) for labels in labels_in_layers]
    n_clusters = max([max(labels) for labels in labels_in_layers]) + 1

    if model_tag == 'vgg16':
        lucid_net = models.VGG16_caffe()
    elif model_tag == 'vgg19':
        lucid_net = models.VGG19_caffe()
    else:
        lucid_net = models.ResnetV1_50_slim()
    lucid_net.load_graphdef()
    layer_map = NETWORK_LAYER_MAP[model_tag]

    max_images = []  # to be filled with images that maximize cluster activations
    max_idxs = []  # to be filled with indices for true sub-cluster neurons
    # min_images = []  # to be filled with images that minimize cluster activations
    random_max_images = []  # to be filled with images that maximize random units activations
    random_max_idxs = []  # to be filled with indices for random sub-cluster neurons
    # random_min_images = []  # to be filled with images that minimize random units activations
    max_losses = []  # to be filled with losses
    # min_losses = []  # to be filled with losses
    random_max_losses = []  # to be filled with losses
    # random_min_losses = []  # to be filled with losses
    sm_sizes = []  # list of submodule sizes
    sm_layer_sizes = []
    sm_layers = []  # list of layer names
    sm_clusters = []  # list of clusters

    for layer_name, labels, layer_size in zip(layer_names, labels_in_layers, layer_sizes):

        if layer_name not in layer_map.keys():
            continue

        lucid_name = layer_map[layer_name]
        max_size = max_prop * layer_size

        for clust_i in range(int(n_clusters)):

            sm_binary = labels == clust_i
            sm_size = sum(sm_binary)
            if sm_size <= min_size or sm_size >= max_size:  # skip if too big or small
                continue

            sm_sizes.append(sm_size)
            sm_layer_sizes.append(layer_size)
            sm_layers.append(layer_name)
            sm_clusters.append(clust_i)

            print(f'{model_tag}, layer names: {layer_name}, {lucid_name}')
            print(f'submodule_size: {sm_size}, layer_size: {layer_size}')

            sm_idxs = [i for i in range(layer_size) if sm_binary[i]]
            max_idxs.append(np.array(sm_idxs))
            max_obj = sum([custom_channel(lucid_name, unit).obj_abs() for unit in sm_idxs])

            max_im, max_loss = render_vis_with_loss(lucid_net, max_obj, size=IMAGE_SIZE_IMAGENET, thresholds=(64,))
            max_images.append(max_im)
            max_losses.append(max_loss)
            # min_im, min_loss = render_vis_with_loss(lucid_net, min_obj)
            # min_images.append(min_im)
            # min_losses.append(min_loss)
            if display:
                print(f'loss: {round(max_loss, 3)}')
                show(max_im)

            rdm_losses = []
            rdm_ims = []
            for _ in range(n_random):  # random max results
                rdm_idxs = np.random.choice(np.array(range(layer_size)), size=sm_size, replace=False)
                random_max_obj = sum([custom_channel(lucid_name, unit).obj_abs() for unit in rdm_idxs])
                random_max_im, random_max_loss = render_vis_with_loss(lucid_net, random_max_obj,
                                                                      size=IMAGE_SIZE_IMAGENET,
                                                                      thresholds=(64,))
                random_max_images.append(random_max_im)
                random_max_losses.append(random_max_loss)
                random_max_idxs.append(rdm_idxs)
                rdm_losses.append(round(random_max_loss, 3))
                rdm_ims.append(np.squeeze(random_max_im))
            if display:
                print(f'random losses: {rdm_losses}')
                show(np.hstack(rdm_ims))

            # for _ in range(n_random):  # random min results
            #     rdm_idxs = np.random.choice(np.array(range(layer_size)), size=sm_size, replace=False)
            #     random_min_obj = -1 * sum([objectives.channel(lucid_name, unit) for unit in rdm_idxs])
            #     random_min_im, random_min_loss = render_vis_with_loss(lucid_net, random_min_obj)
            #     random_min_images.append(random_min_im)
            #     random_min_losses.append(random_min_loss)

    max_images = np.squeeze(np.array(max_images))
    # min_images = np.squeeze(np.array(min_images))
    random_max_images = np.squeeze(np.array(random_max_images))
    # random_min_images = np.squeeze(np.array(random_min_images))
    max_losses = np.array(max_losses)
    # min_losses = np.array(min_losses)
    random_max_losses = np.array(random_max_losses)
    # random_min_losses = np.array(random_min_losses)

    results = {'max_images': max_images,  # 'min_images': min_images,
               'random_max_images': random_max_images,  # 'random_min_images': random_min_images,
               'max_losses': max_losses,  # 'min_losses': min_losses,
               'random_max_losses': random_max_losses,  # 'random_min_losses': random_min_losses,
               'sm_sizes': sm_sizes, 'sm_layer_sizes': sm_layer_sizes,
               'sm_layers': sm_layers, 'sm_clusters': sm_clusters,
               'max_idxs': max_idxs, 'random_max_idxs': random_max_idxs}

    sfx = '_max_data.pkl'
    sfx = '_local' + sfx if local else sfx
    sfx = '_act' + sfx if use_activations else sfx
    with open(savedir + model_tag + sfx, 'wb') as f:
        pickle.dump(results, f, protocol=4)  # protocol 4 for large objects


def evaluate_imagenet_visualizations(model_tag, use_activations=False, local=False,
                                     min_size=MIN_SIZE, max_prop=MAX_PROP,
                                     infodir='/project/nn_clustering/results/',
                                     data_dir='/project/nn_clustering/datasets/'):

    assert model_tag in VIS_NETS

    sfx = '_max_data.pkl'
    sfx = '_local' + sfx if local else sfx
    sfx = '_act' + sfx if use_activations else sfx
    with open(data_dir + model_tag + sfx, 'rb') as f:
        data = pickle.load(f)

    # unpack data
    max_images = data['max_images']
    max_idxs = data['max_idxs']
    # min_images = data['min_images']
    random_max_images = data['random_max_images']
    random_max_idxs = data['random_max_idxs']
    # random_min_images = data['random_min_images']
    max_losses = data['max_losses']
    # min_losses = data['min_losses']
    random_max_losses = data['random_max_losses']
    # random_min_losses = data['random_min_losses']
    sm_sizes = data['sm_sizes']
    sm_layers = data['sm_layers']
    sm_layer_sizes = data['sm_layer_sizes']
    sm_clusters = data['sm_clusters']
    n_examples = len(sm_sizes)
    n_random = int(len(random_max_images) / n_examples)
    input_side = max_images.shape[1]

    # get model
    net, preprocess = Classifiers.get(model_tag)  # get network object and preprocess fn
    model = net((input_side, input_side, 3), weights='imagenet')  # get network tf.keras.model

    # get predictions
    max_preds = model.predict(max_images)
    # min_preds = model.predict(min_images)
    random_max_preds = np.reshape(model.predict(random_max_images), (n_examples, n_random, -1))
    # random_min_preds = np.reshape(model.predict(random_min_images), (n_examples, n_random, -1))

    # get entropies
    max_entropies = np.array([entropy(pred) for pred in max_preds])
    # min_entropies = np.array([entropy(pred) for pred in min_preds])
    random_max_entropies = np.array([[entropy(pred) for pred in reps] for reps in random_max_preds])
    # random_min_entropies = np.array([[entropy(pred) for pred in reps] for reps in random_min_preds])

    # get dispersions
    # max_dispersions = np.zeros_like(max_entropies)
    # max_covs = np.zeros_like(max_entropies)
    # random_max_dispersions = np.zeros_like(random_max_entropies)

    with open(infodir + model_tag + '_clustering_info.pkl', 'rb') as f:
        clustering_info = pickle.load(f)

    layer_names = clustering_info['layers']
    labels_in_layers = [np.array(lyr_labels) for lyr_labels in clustering_info['labels']]
    layer_sizes = [len(labels) for labels in labels_in_layers]
    n_clusters = max([max(labels) for labels in labels_in_layers]) + 1
    layer_map = NETWORK_LAYER_MAP[model_tag]

    example_i = 0
    for layer_name, labels, layer_size in zip(layer_names, labels_in_layers, layer_sizes):
        if layer_name not in layer_map.keys():
            continue
        max_size = max_prop * layer_size
        for clust_i in range(n_clusters):
            sm_binary = labels == clust_i
            sm_size = sum(sm_binary)
            if sm_size <= min_size or sm_size >= max_size:  # skip if too big or small
                continue

            inp = model.input  # input placeholder
            outputs = [layer.output for layer in model.layers if layer.name==layer_name]
            functor = tf.keras.backend.function([inp, tf.keras.backend.learning_phase()], outputs)
            activations = np.squeeze(functor([np.expand_dims(max_images[example_i], 0), 0]))
            activations = np.array([np.linalg.norm(activations[:, :, i], 1)
                                    for i in range(activations.shape[2])])
            activations[activations < 0] = 0
            # true_std = np.std(activations[max_idxs[example_i]])
            # true_cov = true_std / np.mean(activations[max_idxs[example_i]])
            # all_random_stds = []
            for rdm_i in range(n_random):
                rdm_activations = np.squeeze(
                    functor([np.expand_dims(random_max_images[n_random * example_i + rdm_i], 0), 0]))
                rdm_activations = np.array([np.linalg.norm(rdm_activations[:, :, i], 1)
                                            for i in range(rdm_activations.shape[2])])
                rdm_activations[rdm_activations < 0] = 0  # relu
                # all_random_stds.append(np.std(rdm_activations[random_max_idxs[n_random * example_i + rdm_i]]))

            # max_dispersions[example_i] = true_std
            # max_covs[example_i] = true_cov
            # random_max_dispersions[example_i] = np.array(all_random_stds)
            example_i += 1

    # reshape losses
    random_max_losses = np.reshape(random_max_losses, (n_examples, n_random))

    # get percentiles
    max_percentiles_entropy = np.array([compute_pvalue(max_entropies[i], random_max_entropies[i])
                                        for i in range(len(max_entropies))])
    max_percentiles_loss = np.array([compute_pvalue(max_losses[i], random_max_losses[i], side='right')
                                     for i in range(len(max_losses))])

    # get effect sizes
    effect_factors_entropy = np.nan_to_num(np.array([np.mean(max_entropies[i]) /
                                              (random_max_entropies[i] + np.mean(max_entropies[i]))
                                              for i in range(len(max_entropies))]))
    effect_factors_loss = np.nan_to_num(np.array([np.mean(max_losses[i]) /
                                           (np.mean(max_losses[i]) + random_max_losses[i])
                                           for i in range(len(max_losses))]))

    # get pvalues
    max_chi2_p_entropy = chi2_categorical_test(max_percentiles_entropy, n_random)
    max_fisher_p_entropy = combine_ps(max_percentiles_entropy, n_random)
    fisher_stat_entropy = fisher_stat(max_percentiles_entropy, n_random)
    max_chi2_p_loss = chi2_categorical_test(max_percentiles_loss, n_random)
    max_fisher_p_loss = combine_ps(max_percentiles_loss, n_random)
    fisher_stat_loss = fisher_stat(max_percentiles_loss, n_random)

    results = {'percentiles': (max_percentiles_entropy, max_percentiles_loss,),
               'effect_factors': (effect_factors_entropy, effect_factors_loss,),
               'chi2_ps': (max_chi2_p_entropy, max_chi2_p_loss,),
               'fisher_ps': (max_fisher_p_entropy, max_fisher_p_loss,),
               'fisher_stats': (fisher_stat_entropy, fisher_stat_loss,),
               'sm_layers': sm_layers, 'sm_sizes': sm_sizes,
               'sm_layer_sizes': sm_layer_sizes, 'sm_clusters': sm_clusters}

    return results
