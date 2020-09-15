import tensorflow as tf
from absl import app, flags
import math
import os
from pathlib import Path
import matplotlib.pyplot as plt


flags.DEFINE_string('config', 'config.yaml', 'path to config file')
flags.DEFINE_string('image_dir', r'..\dataset\flower_photos', 'path to images dir')
flags.DEFINE_list('labels', None, 'list of class label')

FLAGS = flags.FLAGS

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def data_gen(itr):
    for x, y in itr:
        yield (x, y.argmax(axis=1))


def main(argv):
    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        validation_split=0.1
    )
    itr_train = tf.keras.preprocessing.image.DirectoryIterator(
        FLAGS.image_dir,
        img_gen,
        (224, 224),
        color_mode='rgb',
        data_format='channels_last',
        batch_size=8,
        interpolation='bicubic',
        subset='training',
        classes=FLAGS.labels
    )
    itr_val = tf.keras.preprocessing.image.DirectoryIterator(
        FLAGS.image_dir,
        img_gen,
        (224, 224),
        color_mode='rgb',
        data_format='channels_last',
        batch_size=8,
        interpolation='bicubic',
        subset='validation',
        classes=FLAGS.labels
    )

    def get_1st_dir(root, path):
        root = Path(root)
        path = Path(path)
        rel_path = path.relative_to(root)
        return str(rel_path).split(os.path.sep)[0]

    labels = [get_1st_dir(FLAGS.image_dir, x) for x in itr_train.filepaths]
    labels = sorted(set(labels), key=labels.index)
    ids = sorted(set(itr_train.labels), key=list(itr_train.labels).index)

    labels_map = {}
    for label, id in zip(labels, ids):
        labels_map[label] = id

    print(labels_map)

    for i, data in enumerate(itr_train[0][0]):
        print(f'tmp{i:04d}.png')
        plt.imsave(f'tmp{i:04d}.png', data/255)

    tf.keras.backend.set_image_data_format('channels_last')
    tf.keras.backend.set_learning_phase(1)

    model = tf.keras.applications.MobileNet(input_shape=(224, 224, 3),
                                            weights=None, classes=len(ids))
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    print(itr_train[0][0].shape, itr_train[0][1].shape)

    history = model.fit(
        data_gen(itr_train),
        epochs=1,
        verbose=1,
        steps_per_epoch=math.ceil(itr_train.samples/itr_train.batch_size),
        validation_data=data_gen(itr_val),
        validation_steps=math.ceil(itr_val.samples/itr_val.batch_size)
    )

    print(history.history)

    return 0


if __name__ == '__main__':
    app.run(main)
