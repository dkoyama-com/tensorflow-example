import tensorflow as tf
from absl import app, flags
import os
import yaml
import sys
import math
import dataset_factory
import numpy as np
import matplotlib.pyplot as plt


flags.DEFINE_string('config', 'config.yaml', 'path to config file')

FLAGS = flags.FLAGS

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(argv):

    seed = None

    if os.path.exists(FLAGS.config):
        with open(FLAGS.config) as ifp:
            cfg = yaml.safe_load(ifp)

            augmentations = cfg['dataset']['train']['augmentations'] if 'augmentations' in cfg['dataset']['train'] else []
            train_dataset = dataset_factory.get_dataset(cfg['dataset']['dataset_dir'], cfg['dataset']['train']['pattern'],
                                                        cfg['training']['batch_size'],
                                                        cfg['dataset']['height'], cfg['dataset']['width'],
                                                        cfg['dataset']['channel'],
                                                        augmentations, seed,
                                                        cfg['dataset']['mul'], cfg['dataset']['add'])

            augmentations = cfg['dataset']['validation']['augmentations'] if 'augmentations' in cfg['dataset']['validation'] else []
            val_dataset = dataset_factory.get_dataset(cfg['dataset']['dataset_dir'], cfg['dataset']['validation']['pattern'],
                                                      cfg['validation']['batch_size'],
                                                      cfg['dataset']['height'], cfg['dataset']['width'],
                                                      cfg['dataset']['channel'],
                                                      augmentations, seed,
                                                      cfg['dataset']['mul'], cfg['dataset']['add'])
#           for data in train_dataset.take(1):
#               print(data[0].shape)
#               print(data[0][0].shape)
#               img = data[0][0].numpy()
#               print(np.min(img), np.max(img))
#               plt.imsave('tmp.png', img)

            tf.keras.backend.set_image_data_format('channels_last')
            tf.keras.backend.set_learning_phase(1)

            num_class = cfg['dataset']['num_class']
            height = cfg['dataset']['height']
            width = cfg['dataset']['width']
            channel = cfg['dataset']['channel']

            if cfg['model']['name'] == 'mobilenet':

                alpha = cfg['model']['alpha']
                depth_multiplier = cfg['model']['depth_multiplier']
                dropout = cfg['model']['dropout']

                model = tf.keras.applications.MobileNet(input_shape=(height, width, channel),
                                                        alpha=alpha, depth_multiplier=depth_multiplier,
                                                        include_top=False, pooling='avg',
                                                        weights='imagenet', classes=num_class)

                inputs = model.get_layer('input_1').input
                outputs = model.get_layer('global_average_pooling2d').output
                outputs = tf.keras.layers.Reshape((1, 1, int(1024 * alpha)), name='reshape_1')(outputs)
                outputs = tf.keras.layers.Dropout(dropout, name='dropout')(outputs)
                outputs = tf.keras.layers.Conv2D(num_class, (1, 1), padding='same', name='conv_preds')(outputs)
                outputs = tf.keras.layers.Reshape((num_class,), name='reshape_2')(outputs)
                outputs = tf.keras.layers.Activation('softmax', name='act_softmax')(outputs)

                model = tf.keras.Model(inputs, outputs, name='mobilenet_%0.2f_%s_%d' % (alpha, height, num_class))

            else:
                print('Error: not supported model', file=sys.stderr)
                return 1

            model.summary()

            if cfg['optimizer']['name'] == 'adam':

                learning_rate = cfg['optimizer']['learning_rate']
                beta_1 = cfg['optimizer']['beta_1']
                beta_2 = cfg['optimizer']['beta_2']
                epsilon = cfg['optimizer']['epsilon']
                amsgrad = cfg['optimizer']['amsgrad']

                optimizer = tf.keras.optimizers.Adam(
                    learning_rate,
                    beta_1, beta_2, epsilon,
                    amsgrad
                )
            else:
                print('Error: not supported optimizer', file=sys.stderr)
                return 1

            if cfg['loss']['name'] == 'sparse_categorical_cross_entropy':

                loss = tf.keras.losses.SparseCategoricalCrossentropy()

            else:
                print('Error: not supported optimizer', file=sys.stderr)
                return 1

            metrics = []
            for metric_config in cfg['metrics']:
                if metric_config['name'] == 'sparse_categorical_accuracy':
                    metrics.append(tf.keras.metrics.SparseCategoricalAccuracy())

            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics
            )

            if not os.path.exists(cfg['save']['dir_path']):
                os.makedirs(cfg['save']['dir_path'])

            cbs = [
                tf.keras.callbacks.TensorBoard(
                    log_dir=cfg['save']['dir_path'],
                    write_graph=cfg['summary']['write_graph'],
                    write_images=cfg['summary']['write_images'],
                    update_freq=cfg['summary']['update_freq'],
                    histogram_freq=cfg['summary']['histogram_freq']
                )
            ]

            epoch = 0
            history = model.fit(
                train_dataset,
                initial_epoch=epoch,
                epochs=cfg['training']['epochs'],
                verbose=cfg['training']['verbose'],
                callbacks=cbs,
                validation_data=val_dataset,
                steps_per_epoch=math.ceil(cfg['dataset']['train']['count'] / cfg['training']['batch_size']),
                validation_steps=cfg['validation']['steps'],
                validation_freq=cfg['validation']['freq']
            )
            epoch += cfg['training']['epochs']

            print(history.history)

            results = model.evaluate(
                val_dataset,
                verbose=cfg['validation']['verbose'],
                steps=math.ceil(cfg['dataset']['validation']['count'] / cfg['validation']['batch_size'])
            )
            print("test loss, test acc:", results)

            fname = os.path.join(cfg['save']['dir_path'],
                                 f'{cfg["save"]["basename"]}-{epoch:06d}.{"h5" if cfg["save"]["format"]=="h5" else ""}')

            model.save(fname, save_format=cfg['save']['format'])

            return 0

    print('Error: config file not found', file=sys.stderr)
    return 1


if __name__ == '__main__':
    app.run(main)
