# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Instead of using tf.data.Dataset.from_tensor_slices, we use tf.data.Dataset.from_generator
import numpy as np
import tensorflow as tf

def pandas_input_fn_mod(
    df, y_col=None, batch_size=128, num_epochs=1, shuffle=False, seed=None, num_cate_features=36
):
    """Pandas input function for TensorFlow high-level API Estimator.
    This function returns a `tf.data.Dataset` function.

    .. note::

        `tf.estimator.inputs.pandas_input_fn` cannot handle array/list column properly.
        For more information, see https://www.tensorflow.org/api_docs/python/tf/estimator/inputs/numpy_input_fn

    Args:
        df (pandas.DataFrame): Data containing features.
        y_col (str): Label column name if df has it.
        batch_size (int): Batch size for the input function.
        num_epochs (int): Number of epochs to iterate over data. If `None`, it will run forever.
        shuffle (bool): If True, shuffles the data queue.
        seed (int): Random seed for shuffle.

    Returns:
        tf.data.Dataset: Function.
    """

    X_df = df.copy()
    if y_col is not None:
        y = X_df.pop(y_col).values
    else:
        y = None

    X = {}
    for col in X_df.columns:
        values = X_df[col].values
        if isinstance(values[0], (list, np.ndarray)):
            values = np.array(values.tolist(), dtype=np.float32)
        X[col] = values

    return lambda: _dataset_mod(
        x=X,
        y=y,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle,
        seed=seed,
        num_cate_features=num_cate_features,
    )


def _dataset_mod(x, y=None, batch_size=128, num_epochs=1, shuffle=False, seed=None, num_cate_features=36):
    if y is None:
        dataset = tf.data.Dataset.from_tensor_slices(x)
    else:
        # dataset = tf.data.Dataset.from_tensor_slices((x, y))
        def genenerator():
            for u, i, g, l in zip(x['userID'], x['itemID'], x['genre'], y):
                my_features = {}
                my_features['userID'] = u
                my_features['itemID'] = i
                my_features['genre'] = g
                yield my_features, l # TF takes care of repeat!
        dataset = tf.data.Dataset.from_generator(genenerator, output_signature=(
                                                 {'userID': tf.TensorSpec(shape=(), dtype=tf.int64),
                                                  'itemID': tf.TensorSpec(shape=(), dtype=tf.int64),
                                                  'genre': tf.TensorSpec(shape=(num_cate_features), dtype=tf.float32)},
                                                 tf.TensorSpec(shape=(), dtype=tf.float64)))
    if shuffle:
        dataset = dataset.shuffle(
            1000, seed=seed, reshuffle_each_iteration=True  # buffer size = 1000
        )
    elif seed is not None:
        import warnings

        warnings.warn("Seed was set but `shuffle=False`. Seed will be ignored.")

    return dataset.repeat(num_epochs).batch(batch_size)
