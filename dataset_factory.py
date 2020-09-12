import tensorflow as tf
import glob
import os
from typing import Union, List, Dict, Any


feature_description = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=""),
    'image/format': tf.io.FixedLenFeature([], tf.string, default_value=b'jpeg'),
    'image/class/label': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    'image/height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image/width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
}


def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)


def _preprocess_image(decoded: tf.data.Dataset,
                      input_height: int, input_width: int, channel: int,
                      augmentations: List[Union[str, Dict[str, Any]]], seed: Union[int, None],
                      height: int, width: int) -> tf.data.Dataset:
    img = tf.reshape(decoded, tf.stack([input_height, input_width, channel]))
    img = tf.image.convert_image_dtype(img, tf.float32)

    for augmentation in augmentations:
        if 'random_brightness' in augmentation:
            max_delta = 0.3
            if type(augmentation['random_brightness']) is dict and 'max_delta' in augmentation['random_brightness']:
                max_delta = augmentation['random_brightness']['max_delta']
            img = tf.image.random_brightness(img, max_delta, seed)

        if 'random_contrast' in augmentation:
            lower = 0.8
            if type(augmentation['random_contrast']) is dict and 'lower' in augmentation['random_contrast']:
                lower = augmentation['random_contrast']['lower']
            upper = 1.2
            if type(augmentation['random_contrast']) is dict and 'upper' in augmentation['random_contrast']:
                upper = augmentation['random_contrast']['upper']
            img = tf.image.random_contrast(img, lower, upper, seed)

        if 'random_crop' in augmentation:
            min_h = int(tf.cast(input_height, tf.float32) * 0.5)
            if type(augmentation['random_crop']) is dict and 'min_h_ratio' in augmentation['random_crop']:
                min_h = int(tf.cast(input_height, tf.float32) * augmentation['random_crop']['min_h_ratio'])
            min_w = int(tf.cast(input_width, tf.float32) * 0.5)
            if type(augmentation['random_crop']) is dict and 'min_w_ratio' in augmentation['random_crop']:
                min_w = int(tf.cast(input_width, tf.float32) * augmentation['random_crop']['min_w_ratio'])
            img = tf.image.random_crop(img, [min_h, min_w, channel], seed)

        if 'random_flip_left_right' in augmentation:
            img = tf.image.random_flip_left_right(img, seed)

        if 'random_flip_up_down' in augmentation:
            img = tf.image.random_flip_up_down(img, seed)

        if 'random_hue' in augmentation:
            max_delta = 0.3
            if type(augmentation['random_hue']) is dict and 'max_delta' in augmentation['random_hue']:
                max_delta = augmentation['random_hue']['max_delta']
            img = tf.image.random_hue(img, max_delta, seed)

    img = tf.image.resize(img, [height, width])
    img = tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0)

    return img


def _preprocess_jpeg(encoded: tf.data.Dataset,
                     input_height: int, input_width: int, channel: int,
                     augmentations: List[str], seed: Union[int, None],
                     height: int, width: int) -> tf.data.Dataset:
    decoded = tf.io.decode_jpeg(encoded, channel)
    return _preprocess_image(decoded,
                             input_height, input_width, channel,
                             augmentations, seed, height, width)


def _preprocess_png(encoded: tf.data.Dataset,
                    input_height: int, input_width: int, channel: int,
                    augmentations: List[str], seed: Union[int, None],
                    height: int, width: int) -> tf.data.Dataset:
    decoded = tf.io.decode_png(encoded, channel)
    return _preprocess_image(decoded,
                             input_height, input_width, channel,
                             augmentations, seed, height, width)


def get_dataset(dataset_dir: str, file_pattern: str = '*',
                batch_size: int = 1,
                height: Union[int, None] = None, width: Union[int, None] = None,
                channel: int = 3,
                augmentations: List[Union[str, Dict[str, Any]]] = [], seed: Union[int, None] = None) -> tf.data.Dataset:
    """指定したディレクトリ内の TFRecord ファイルから Dataset を生成する

    :param dataset_dir: TFRecord ファイルを探索するディレクトリ
    :type dataset_dir: str, optional
    :param file_pattern: パターンにマッチするファイル名の TFRecord のみ使用する, defaults to '*'
    :type file_pattern: str, optional
    :param batch_size: データセットのバッチサイズ, defaults to 1
    :type batch_size: int, optional
    :param height: 画像データの高さ, defaults to None
    :type height: Union[int, None], optional
    :param width: 画像データの幅, defaults to None
    :type width: Union[int, None], optional
    :param channel: 入力画像データのチャンネル数, defaults to 3
    :type channel: int, optional
    :param augmentations: Augumentation　の設定リスト, defaults to []
    :type augmentations: List[Union[str, Dict[str, Any]]], optional
    :param seed: 乱数のシード, defaults to 0
    :type seed: int, optional
    :return: 生成したデータセット
    :rtype: tf.data.Dataset
    """
    files = glob.glob(os.path.join(dataset_dir, file_pattern))
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x))
    dataset = dataset.map(_parse_function)
    dataset_jpeg = dataset.filter(lambda x: x['image/format'] == b'jpg')
    images_jpeg = dataset_jpeg.map(lambda x: _preprocess_jpeg(x['image/encoded'],
                                                              x['image/height'], x['image/width'], channel,
                                                              augmentations, seed, height, width))
    dataset_png = dataset.filter(lambda x: x['image/format'] == b'png')
    images_png = dataset_png.map(lambda x: _preprocess_png(x['image/encoded'],
                                                           x['image/height'], x['image/width'], channel,
                                                           augmentations, seed, height, width))
    images = tf.data.Dataset.concatenate(images_jpeg, images_png)
    images = images.batch(batch_size)

    labels_jpeg = dataset_jpeg.map(lambda x: x['image/class/label'])
    labels_png = dataset_png.map(lambda x: x['image/class/label'])
    labels = tf.data.Dataset.concatenate(labels_jpeg, labels_png)
    labels = labels.batch(batch_size)

    dataset = tf.data.Dataset.zip((images, labels))
    dataset = dataset.shuffle(batch_size, seed, True)
    dataset = dataset.repeat()

    return dataset
