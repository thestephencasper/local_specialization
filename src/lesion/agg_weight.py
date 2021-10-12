import numpy as np

from src.lesion.experimentation import _single_damaged_neurons_gen, _layers_labels_gen

DIRECTIONS = ('inbound', 'outbound')


# Understanding Flatten
"""
import tensorflow as tf
from tensorflow import keras

flt = keras.layers.Flatten()

n_neurons = 3

# default: channel last!
inp = np.arange(1*7*7*n_neurons).reshape(1, 7, 7, n_neurons)


with tf.Session() as s:
    outp = s.run(flt(inp))
    
    
neuron_id = 2
assert (outp[0, np.arange(neuron_id, 7*7*n_neurons, n_neurons)]
        == inp[0, :, :, neuron_id].flatten()).all()
"""

def conv2_agg(w): return np.abs(w).mean(axis=(0, 1)).mean()
def fc_agg(w): return np.abs(w).mean()

def aggregate_weights(layer_id, neuron_ids, direction, network_type, weights):

    # In this function, `layer_id` is aligned to the indexing
    # of the weights from `extract_weights` and not according
    # to the layer_label generators (e.g., _layers_labels_gen)
    
    # cnn kernel shape: [filter_height, filter_width, in_channels, out_channels]
    # fc weight shape: [in, out]
    
    assert direction in DIRECTIONS
    
    # same layer
    if direction == 'inbound':  # to the current layer
        if len(weights[layer_id].shape) == 4:  # CNN
            assert network_type  == 'cnn'
            return conv2_agg(weights[layer_id][:, :, :, neuron_ids])
        else:  # FC
            assert network_type  == 'mlp'
            return fc_agg(weights[layer_id][:, neuron_ids])
    
    elif direction == 'outbound': # next layer = coming out of the previous layers  
        if len(weights[layer_id+1].shape) == 4:  # CNN
            assert network_type  == 'cnn'
            return conv2_agg(weights[layer_id+1][:, :, neuron_ids, :])

        else:  # FC
            if network_type == 'mlp':
                return fc_agg(weights[layer_id+1][neuron_ids, :])

            if network_type == 'cnn':
                raise NotImplementedError()
                
                assert len(weights[layer_id].shape) == 4  # conv2d

                n_filters = weights[layer_id].shape[3]
                filter_size = weights[layer_id].shape[0] * weights[layer_id].shape[1]
                print(n_filters, filter_size)
                
                # This part is very tricky
                # We need to figure out which neurons in the "in" side
                # of the FC layer correspond to the neurons in the previous
                # Conv2D layer. 
                #
                # 1. In Conv2D, the default dims order is channels_last:
                #    (batch, height, width, channels)
                #
                # 2. Flatten works from the last dim:
                #    [(0,0,0), (0,0,1), ..., (0, 1, 0), (0, 1, 1), ...]
                #
                # 3. So we need to collect the FC "incoming" weights
                #    in jumps of height*width to collect the outbound
                #    weights of a given neruon_id in the Conv2D layer
                
                # NOTE: IT STILL DON'T WORK BECAUSE OF THE max_pool LAYER!
                fc_neurons_ids = [np.arange(neuron_id, filter_size*n_filters, n_filters)
                                  for neuron_id in neuron_ids]
                expanded_neuron_ids = np.concatenate(fc_neurons_ids)

                return fc_agg(weights[layer_id+1][expanded_neuron_ids, :])


def agg_weights_by_subcluster(layer_id, label_id, neuron_ids, network_type, weights):
    # The iterator enumerate layers from 1, and not from 0.
    # So we need layer_id-1 to pull out the correct weight from the model
    d = {'layer': layer_id,
             'label': label_id,
             'inbound': aggregate_weights(layer_id-1, neuron_ids, 'inbound', network_type, weights)}

    try:
        d['outbound'] = aggregate_weights(layer_id-1, neuron_ids, 'outbound', network_type, weights)
    except NotImplementedError:
        pass
    return d
            
def agg_weights_by_partitioning(labels, network_type, weights, layer_widths):
    subcluster_iter = _single_damaged_neurons_gen(
                      _layers_labels_gen(network_type,
                                                         layer_widths,
                                                         labels, None))
    return [agg_weights_by_subcluster(layer_id, label_id, neuron_ids, network_type, weights)
            for layer_id, label_id, neuron_ids, _ in subcluster_iter]

def test_agg_weights_by_partitioning(network_type):
    
    from tensorflow.keras import datasets, layers, models
    from src.utils import extract_weights
    from src.visualization import extract_layer_widths


    model = models.Sequential()
    
    if network_type == 'mlp':
        labels = np.concatenate([np.full(123, -1),                 # input
                                 np.full(16, 1), np.full(16, 2),   # first fc
                                 np.full(32, 3), np.full(32, 4),   # second fc
                                 np.full( 8, 5), np.full( 8, 6)])  # third fc

        model.add(layers.Dense(32, input_shape=(123,)))
        model.add(layers.Dense(64))
        model.add(layers.Dense(16))
        model.add(layers.Dense(2))
    else:
        labels = np.concatenate([np.full(16, 1), np.full(16, 2),   # first cov2d
                                 np.full(32, 3), np.full(32, 4),   # second cov2d
                                 np.full( 8, 5), np.full( 8, 6)])  # third cov2d

        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(16, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(2, activation='softmax'))

            
    weights, _ = extract_weights(model, with_bias=True)

    if network_type == 'mlp':
        layer_widths = extract_layer_widths(weights)
        
        results = agg_weights_by_partitioning(labels, network_type,
                                              weights, layer_widths)

        assert len(results) == 6
    
        np.testing.assert_almost_equal(results[0]['inbound'],
                                       np.abs(model.get_weights()[0][:, :16]).mean())
        np.testing.assert_almost_equal(results[1]['inbound'],
                               np.abs(model.get_weights()[0][:, 16:]).mean())
        np.testing.assert_almost_equal(results[2]['inbound'],
                               np.abs(model.get_weights()[2][:, :32]).mean())
        np.testing.assert_almost_equal(results[3]['inbound'],
                               np.abs(model.get_weights()[2][:, 32:]).mean())
        np.testing.assert_almost_equal(results[4]['inbound'],
                               np.abs(model.get_weights()[4][:, :8]).mean())
        np.testing.assert_almost_equal(results[5]['inbound'],
                               np.abs(model.get_weights()[4][:, 8:]).mean())

        np.testing.assert_almost_equal(results[0]['outbound'],
                               np.abs(model.get_weights()[2][:16, :]).mean())
        np.testing.assert_almost_equal(results[1]['outbound'],
                               np.abs(model.get_weights()[2][16:, :]).mean())
        np.testing.assert_almost_equal(results[2]['outbound'],
                               np.abs(model.get_weights()[4][:32, :]).mean())
        np.testing.assert_almost_equal(results[3]['outbound'],
                               np.abs(model.get_weights()[4][32:, :]).mean())
        np.testing.assert_almost_equal(results[4]['outbound'],
                               np.abs(model.get_weights()[6][:8, :]).mean())
        np.testing.assert_almost_equal(results[5]['outbound'],
                               np.abs(model.get_weights()[6][8:, :]).mean())
        
    else:
        layer_widths = []
        weight_shapes = [layer_weights.shape for layer_weights in weights]
        n_conv = sum(len(ws) == 4 for ws in weight_shapes)
        layer_widths.extend([weight_shapes[i][-1] for i in range(n_conv)])
        layer_widths.extend([ws[-1] for ws in weight_shapes[n_conv:]])

        weights = weights[:n_conv+1]
        layer_widths = layer_widths[:n_conv+1]

        results = agg_weights_by_partitioning(labels, network_type,
                                              weights, layer_widths)
        assert len(results) == 6
        assert 'outbound' not in results[4] and 'outbound' not in results[5]
        
        np.testing.assert_almost_equal(results[0]['inbound'],
                                       np.abs(model.get_weights()[0][:, :, :, :16]).mean())
        np.testing.assert_almost_equal(results[1]['inbound'],
                               np.abs(model.get_weights()[0][:, :, :, 16:]).mean())
        np.testing.assert_almost_equal(results[2]['inbound'],
                               np.abs(model.get_weights()[2][:, :, :, :32]).mean())
        np.testing.assert_almost_equal(results[3]['inbound'],
                               np.abs(model.get_weights()[2][:, :, :, 32:]).mean())
        np.testing.assert_almost_equal(results[4]['inbound'],
                               np.abs(model.get_weights()[4][:, :, :, :8]).mean())
        np.testing.assert_almost_equal(results[5]['inbound'],
                               np.abs(model.get_weights()[4][:, :, :, 8:]).mean())

        np.testing.assert_almost_equal(results[0]['outbound'],
                               np.abs(model.get_weights()[2][:, :, :16, :]).mean())
        np.testing.assert_almost_equal(results[1]['outbound'],
                               np.abs(model.get_weights()[2][:, :, 16:, :]).mean())
        np.testing.assert_almost_equal(results[2]['outbound'],
                               np.abs(model.get_weights()[4][:, :, :32, :]).mean())
        np.testing.assert_almost_equal(results[3]['outbound'],
                               np.abs(model.get_weights()[4][:, :, 32:, :]).mean())


test_agg_weights_by_partitioning('cnn')
test_agg_weights_by_partitioning('mlp')