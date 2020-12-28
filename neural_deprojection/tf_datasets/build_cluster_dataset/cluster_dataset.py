"""cluster_dataset dataset."""

import tensorflow_datasets as tfds
import os
import tensorflow as tf
import h5py
from astropy.io import fits
import numpy as np

# TODO(cluster_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(cluster_dataset): BibTeX citation
_CITATION = """
"""


class ClusterDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for cluster_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'gas_pos': tfds.features.Tensor(shape=(111858, 3), dtype=tf.float64),
            'xray': tfds.features.Tensor(shape=(4880, 4880), dtype=tf.float64),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('gas_pos', 'xray'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    
    # TODO: (note from Matthijs) Fill in the path for the data directory (where you unzip the data file)
    data_dir = ""
    path_to_dataset = data_dir + '/data'

    return {
        'train': self._generate_examples(path_to_dataset),
    }

  def _generate_examples(self, path):
    clusters = tf.io.gfile.listdir(path + '/clusters_032')
    xrays = tf.io.gfile.listdir(path + '/xray_img_032')

    for i in range(len(clusters)):
      f = h5py.File(os.path.join(path + '/clusters_032', clusters[i]))
      hdul = fits.open(os.path.join(path + '/xray_img_032', xrays[i]))
      record = {
          "gas_pos": np.array(f['PartType0/Coordinates'][()]),
          "xray": np.array(hdul[0].data, dtype='float64'),
      }
      yield i, record
