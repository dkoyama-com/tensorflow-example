import tensorflow as tf
from absl import app, flags
import os


flags.DEFINE_string(
    'model',
    'models\\model.h5',
    'model file'
)

FLAGS = flags.FLAGS

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(argv):
    model = tf.keras.models.load_model(FLAGS.model)
    model.summary()

    return 0


if __name__ == '__main__':
    app.run(main)
