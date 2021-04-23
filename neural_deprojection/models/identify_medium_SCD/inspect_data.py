import os
import glob
import tensorflow as tf

import sys

sys.path.insert(1, '/data/s1825216/git/neural_deprojection/')

from neural_deprojection.models.identify_medium_SCD.generate_data import generate_data, decode_examples


from functools import partial
from timeit import default_timer
from itertools import product
from graph_nets.graphs import GraphsTuple
from graph_nets.utils_np import graphs_tuple_to_networkxs, networkxs_to_graphs_tuple, get_graph
import numpy as np
import networkx as nx
from networkx.drawing import draw, draw_networkx_edges, draw_networkx_nodes
from tqdm import tqdm
from scipy.optimize import bisect
from scipy.spatial.ckdtree import cKDTree
import matplotlib
from scipy import interpolate

matplotlib.use('Agg')
import matplotlib.pyplot as plt

def _get_data(dir):
    """
    Should return the information for a single simulation.

    Args:
        dir: directory with sim data.

    Returns:
        positions for building graph
        properties for putting in nodes and aggregating upwards
        image corresponding to the graph
        extra info corresponding to the example

    """

    f = np.load(os.path.join(dir, 'data.npz'))
    positions = f['positions']
    properties = f['properties']
    image = f['proj_image']
    image = image.reshape((256, 256, 1))

    # properties = properties / np.std(properties, axis=0)        # normalize values

    # extra_info = f['extra_info']

    return positions, properties, image  # , extra_info


def get_data_image(data_dirs):
    """
    Get information of saved data.

    Args:
        data_dirs: data directories

    Returns:

    """

    image_dir = '/data2/hendrix/projection_images/'

    def data_generator():
        for idx, dir in tqdm(enumerate(data_dirs)):
            print("Generating data from {}".format(dir))
            positions, properties, image = _get_data(dir)
            yield (properties, image, dir)

    data_iterable = iter(data_generator())

    while True:
        try:
            (properties, image, dir) = next(data_iterable)
        except StopIteration:
            break
        print('save image...')
        proj_image_idx = len(glob.glob(os.path.join(image_dir, 'proj_image_*')))
        plt.imsave(os.path.join(image_dir, 'proj_image_{}.png'.format(proj_image_idx)),
                   image[:, :, 0])
        print('saved.')


def plot_from_ds_item(ds_item):
    (graph, image, snapshot, projection) = ds_item
    nx_graph = nx.DiGraph()

    pos = dict()  # for plotting node positions.
    dens_list = []
    x_list = []
    y_list = []
    z_list = []
    for i, n in enumerate(graph.nodes.numpy()):
        nx_graph.add_node(i, features=n)
        pos[i] = n[:2]
        dens_list.append(n[7])
        x_list.append(n[0])
        y_list.append(n[1])
        z_list.append(n[2])

    edgelist = []
    for u, v in zip(graph.senders.numpy(), graph.receivers.numpy()):
        nx_graph.add_edge(u, v, features=np.array([1., 0.]))
        nx_graph.add_edge(v, u, features=np.array([1., 0.]))
        edgelist.append((u, v))
        edgelist.append((v, u))

    nx_graph.graph["features"] = np.array([0.])
    # plotting

    print('len(pos) = {}\nlen(edgelist) = {}'.format(len(pos), len(edgelist)))

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(60, 20))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(40, 20))
    fig.suptitle(f'snapshot: {snapshot}, projection: {projection}', size=40)
    ax1.set_aspect(1)
    ax2.set_aspect(1)
    # ax3.set_aspect(1)

    draw_networkx_nodes(nx_graph, ax=ax1, pos=pos, node_size=10,
         node_color=dens_list, cmap='viridis')

    draw_networkx_edges(nx_graph, ax=ax1, pos=pos,
         edge_color='grey', width=0.1, arrowstyle='-', alpha=1)

    # draw(nx_graph, ax=ax1, pos=pos,
    #      edge_color='red', node_size=10, width=0.1, arrowstyle='-',
    #      node_color=dens_list, cmap='viridis', alpha=0.3)

    ax2.imshow(image.numpy()[:, :, 0].T, origin='lower')
    # ax2.set_xlim(0.7,1.3)
    # ax2.set_ylim(0.7,1.3)
    ax2.axis('off')

    # _x = np.linspace(np.min(x_list), np.max(x_list), 700)
    # _y = np.linspace(np.min(y_list), np.max(y_list), 700)
    # _z = np.linspace(np.min(z_list), np.max(z_list), 700)
    # x, y, z = np.meshgrid(_x, _y, _z, indexing='ij')
    #
    # interp = interpolate.griddata((x_list, y_list, z_list),
    #                               dens_list,
    #                               xi=(x, y, z), fill_value=np.min(dens_list))
    #
    # im = np.sum(interp, axis=2)
    # # im = interp[:,:,350]
    # im = np.trapz(interp, axis=2)

    # H, xedges, yedges = np.histogram2d(x_list, y_list, bins=200, weights=dens_list, range=[[-0.5, 4], [-2,2]])
    # print(H)
    # ax3.imshow(H.T, origin='lower', interpolation='bilinear')    #, extent=(xmin, xmax, ymin, ymax))


    image_dir = '/data2/hendrix/images_tf/'
    # mp_lock.acquire()  # make sure no duplicate files are made / replaced
    # graph_image_idx = len(glob.glob(os.path.join(image_dir, 'graph_image_*')))
    # mp_lock.release()  # make sure no duplicate files are made / replaced

    plt.savefig(os.path.join(image_dir, 'graph_image_{}_{}'.format(snapshot, projection)))


def plot_from_big_ds_item(ds_item):
    print('extracting data...')
    t0 = default_timer()
    (graph, image, snapshot, projection) = ds_item
    print('done!', default_timer() - t0)
    print('to nx_graph...')
    t0 = default_timer()
    nx_graph = nx.DiGraph()

    pos = dict()  # for plotting node positions.
    dens_list_i = []
    mass_list = []
    vol_list = []
    x_list_i = []
    y_list_i = []
    z_list_i = []

    for i, n in enumerate(graph.nodes.numpy()):
        dens_list_i.append(n[7])
        mass_list.append(n[9])
        vol_list.append(n[-1])
        x_list_i.append(n[0])
        y_list_i.append(n[1])
        z_list_i.append(n[2])

    dens_list_i = np.array(dens_list_i)
    top_masses = np.array(dens_list_i) > np.percentile(dens_list_i, 98)
    print(top_masses)
    x_list_i = np.array(x_list_i)
    y_list_i = np.array(y_list_i)
    z_list_i = np.array(z_list_i)
    dens_list = dens_list_i[top_masses]
    x_list = x_list_i[top_masses]
    y_list = y_list_i[top_masses]
    z_list = z_list_i[top_masses]

    pos_max = np.max(np.array([np.max(x_list_i), np.max(y_list_i), np.max(z_list_i)]))
    pos_min = np.min(np.array([np.min(x_list_i), np.min(y_list_i), np.min(z_list_i)]))

    pos_max = 8.681422542110612
    pos_min = -8.681422542110612

    x_list = np.array([(x - pos_min) / (pos_max - pos_min) * 256 for x in x_list])
    y_list = np.array([(y - pos_min) / (pos_max - pos_min) * 256 for y in y_list])
    z_list = np.array([(z - pos_min) / (pos_max - pos_min) * 256 for z in z_list])


    for i, (dens, x, y, z) in enumerate(zip(dens_list, x_list, y_list, z_list)):
        nx_graph.add_node(i, features=[x,y,z,dens])
        pos[i] = [x, y]


    nx_graph.graph["features"] = np.array([0.])
    # plotting

    print('done!', default_timer() - t0)
    print('len(pos) = {}'.format(len(pos)))

    print('plotting...')
    t0 = default_timer()

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(60, 20))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(40, 20))
    fig.suptitle(f'snapshot: {snapshot}, projection: {projection}', size=40)
    ax1.set_aspect(1)
    ax2.set_aspect(1)
    ax1.axis('off')
    ax2.axis('off')

    # draw_networkx_edges(nx_graph, ax=ax1, pos=pos,
    #      edge_color='grey', width=0.1, arrowstyle='-', alpha=0.2)

    ax2.imshow(image.numpy()[:, :, 0].T, origin='lower')
    ax1.imshow(image.numpy()[:, :, 0].T, origin='lower')

    draw_networkx_nodes(nx_graph, ax=ax2, pos=pos, node_size=10,
         node_color=dens_list, cmap='viridis')

    print('done!', default_timer() - t0)
    print('saving...')
    t0 = default_timer()
    image_dir = '/home/s1825216/data/train_data/ClaudeData/images/'
    plt.savefig(os.path.join(image_dir, 'graph_image_{}_{}'.format(snapshot, projection)))
    print('done!', default_timer() - t0)


if __name__ == '__main__':
    # test_train_dir = '/data2/hendrix/ClaudeData/M4r5b/'
    # image_dir = '/data2/hendrix/ClaudeData/M4r5b/images/'
    #
    test_train_dir = '/home/s1825216/data/train_data/ClaudeData/train/'

    tfrecords = glob.glob(os.path.join(test_train_dir, '*.tfrecords'))  # list containing tfrecord files
    print(tfrecords)

    tfrecords = tfrecords[::-1]

    print('making dataset...')
    # Extract the dataset (graph tuple, image, example_idx) from the tfrecords files
    dataset = tf.data.TFRecordDataset(tfrecords).map(partial(decode_examples,
                                                             node_shape=(11,),
                                                             image_shape=(256, 256, 1)))  # (graph, image, idx)

    # dataset = dataset.map(lambda graph_data_dict, img, s, p: (GraphsTuple(globals=None, edges=None, **graph_data_dict),
    #                                                           img, s, p))

    for ds_item in iter(dataset):
        # print(ds_item)
        # plot_from_big_ds_item(ds_item)

        (graph_data_dict, image, snapshot, projection) = ds_item
        nodes = graph_data_dict['nodes']
        for i in range(11): # [6,9,10]:
            print(f'\nmax_{i}', tf.reduce_max(nodes[:, i]))
            print(f'min_{i}', tf.reduce_min(nodes[:, i]))
        break


