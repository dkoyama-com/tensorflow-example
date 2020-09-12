import tensorflow as tf
from absl import app
from absl import flags
import os
import yaml
import sys
import dataset_factory
import numpy as np
import matplotlib.pyplot as plt


flags.DEFINE_string('config', 'config.yaml', 'path to config file')

FLAGS = flags.FLAGS


def main(argv):

    seed = None

    if os.path.exists(FLAGS.config):
        with open(FLAGS.config) as ifp:
            cfg = yaml.safe_load(ifp)

            channel_last = cfg['dataset']['channel_last'] if 'channel_last' in cfg['dataset'] else True

            augmentations = cfg['dataset']['train']['augmentations'] if 'augmentations' in cfg['dataset']['train'] else []
            train_dataset = dataset_factory.get_dataset(cfg['dataset']['dataset_dir'], cfg['dataset']['train']['pattern'],
                                                        cfg['dataset']['train']['batch_size'],
                                                        cfg['dataset']['height'], cfg['dataset']['width'],
                                                        cfg['dataset']['channel'], channel_last,
                                                        augmentations, seed)

            augmentations = cfg['dataset']['validation']['augmentations'] if 'augmentations' in cfg['dataset']['validation'] else []
            val_dataset = dataset_factory.get_dataset(cfg['dataset']['dataset_dir'], cfg['dataset']['validation']['pattern'],
                                                      cfg['dataset']['validation']['batch_size'],
                                                      cfg['dataset']['height'], cfg['dataset']['width'],
                                                      cfg['dataset']['channel'], channel_last,
                                                      augmentations, seed)
            for data in train_dataset.take(1):
                print(data[0].shape)
                print(data[0][0].shape)
                img = data[0][0].numpy()
                print(np.min(img), np.max(img))
                plt.imsave('tmp.png', img)

            if channel_last:
                tf.keras.backend.set_image_data_format('channels_last')
            else:
                tf.keras.backend.set_image_data_format('channels_first')

            return 0

    print('Error: config file not found', file=sys.stderr)
    return 1


if __name__ == '__main__':
    app.run(main)
