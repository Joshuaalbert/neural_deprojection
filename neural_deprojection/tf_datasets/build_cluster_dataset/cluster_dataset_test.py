"""cluster_dataset dataset."""

import tensorflow_datasets as tfds
from . import cluster_dataset


class ClusterDatasetTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for cluster_dataset dataset."""

  DATASET_CLASS = cluster_dataset.ClusterDataset
  SPLITS = {
      'train': 1,  # Number of fake train example
  }


if __name__ == '__main__':
  tfds.testing.test_main()
