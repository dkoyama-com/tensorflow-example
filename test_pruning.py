import tensorflow as tf
import tensorflow_model_optimization as tfmot
from absl import app, flags
import os
import tempfile
import math


flags.DEFINE_string(
    'base_model',
    'models\\mobilenet_v1_224_flowers_20200915.h5',
    'model file to optimize'
)
flags.DEFINE_string(
    'opt_model',
    'models\\optimized.h5',
    'model file optimized'
)
flags.DEFINE_string(
    'image_dir',
    '..\\dataset\\flower_photos',
    'path to image directory'
)
flags.DEFINE_list('labels', None, 'list of class label')


FLAGS = flags.FLAGS

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def apply_pruning_to_dense(layer):
    if isinstance(layer, tf.keras.layers.Conv2D):
        return tfmot.sparsity.keras.prune_low_magnitude(layer)
    return layer


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

    model = tf.keras.models.load_model(FLAGS.base_model)
    model.summary()

    base_eval = model.evaluate(itr_val)
    print('Base [loss, accuracy]:', base_eval)

    model_for_pruning = tf.keras.models.clone_model(
        model,
        clone_function=apply_pruning_to_dense
    )

    model_for_pruning.summary()

    log_dir = tempfile.mkdtemp()
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir)
    ]

    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.CategoricalAccuracy()]

    model_for_pruning.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    history = model_for_pruning.fit(
        itr_train,
        epochs=32,
        verbose=1,
        steps_per_epoch=math.ceil(itr_train.samples/itr_train.batch_size),
        callbacks=callbacks
    )

    print(history.history)

    opt_eval = model_for_pruning.evaluate(itr_val)
    print('Optimized [loss, accuracy]:', opt_eval)

    model_for_pruning.save(FLAGS.opt_model, save_format='h5')

    return 0


if __name__ == '__main__':
    app.run(main)
