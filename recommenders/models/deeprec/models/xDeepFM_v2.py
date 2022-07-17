import numpy as np
import tensorflow as tf

from recommenders.models.deeprec.models.base_model_v2 import BaseModel_v2, _activate, _active_layer


__all__ = ["XDeepFMModel_v2"]
class Embedding(tf.keras.layers.Layer):
    """
    The field embedding layer. MLP requires fixed-length vectors as input.
    This function makes sum pooling of feature embeddings for each field.
    Returns:
        embedding:  The result of field embedding layer, with size of #_fields * #_dim.
    """
    def __init__(self, hparams, initializer, iterator, embed_params, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.hparams = hparams
        self.initializer = initializer
        self.iterator = iterator
        self.embed_params = embed_params
        self.embed_layer_size = hparams.FIELD_COUNT * hparams.dim

    def build(self, input_shape):
        feature_cnt = self.hparams.FEATURE_COUNT
        dim = self.hparams.dim

        self.embedding = tf.Variable(
            name="embedding_layer",
            initial_value=self.initializer(shape=[feature_cnt, dim], dtype=tf.float32)
        )
        self.embed_params.append(self.embedding)

    def call(self, inputs):
        hparams = self.hparams
        fm_sparse_index = tf.sparse.SparseTensor(
            self.iterator.dnn_feat_indices,
            self.iterator.dnn_feat_values,
            self.iterator.dnn_feat_shape,
        )
        fm_sparse_weight = tf.sparse.SparseTensor(
            self.iterator.dnn_feat_indices,
            self.iterator.dnn_feat_weights,
            self.iterator.dnn_feat_shape,
        )
        w_fm_nn_input_orgin = tf.nn.embedding_lookup_sparse(
            params=self.embedding,
            sp_ids=fm_sparse_index,
            sp_weights=fm_sparse_weight,
            combiner="sum",
        )
        embed_out = tf.reshape(
            w_fm_nn_input_orgin, [-1, hparams.dim * hparams.FIELD_COUNT]
        )
        return embed_out

class Linear(tf.keras.layers.Layer):
    """
    The linear part for the model.
    This is a linear regression.
    Returns:
        object: Prediction score made by linear regression.
    """
    def __init__(self, hparams, initializer, iterator, layer_params, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.hparams = hparams
        self.initializer = initializer
        self.iterator = iterator
        self.layer_params = layer_params

    def build(self, input_shape):
        feature_cnt = self.hparams.FEATURE_COUNT
        self.linear_w = tf.Variable(
            name="w", initial_value=self.initializer(shape=[feature_cnt, 1], dtype=tf.float32)
        )
        self.linear_b = tf.Variable(
            name="b", initial_value=tf.zeros_initializer()(shape=[1], dtype=tf.float32)
        )
        self.layer_params.append(self.linear_w)
        self.layer_params.append(self.linear_b)

    def call(self, inputs):
        x = tf.sparse.SparseTensor(
            self.iterator.fm_feat_indices,  # indices
            self.iterator.fm_feat_values,   # values
            self.iterator.fm_feat_shape,    # dense shape
        )
        linear_output = tf.add(tf.sparse.sparse_dense_matmul(x, self.linear_w), self.linear_b)

        tf.summary.histogram("linear_part/w", self.linear_w)
        tf.summary.histogram("linear_part/b", self.linear_b)
        return linear_output

class FM(tf.keras.layers.Layer):
    """
    The factorization machine part for the model.
    This is a traditional 2-order FM module.
    Returns:
        object: Prediction score made by factorization machine.
    """
    def __init__(self, _embedding, iterator, **kwargs):
        super(FM, self).__init__(**kwargs)
        self._embedding = _embedding
        self.iterator = iterator

    def call(self, inputs):
        embedding = self._embedding.embedding
        x = tf.sparse.SparseTensor(
            self.iterator.fm_feat_indices,
            self.iterator.fm_feat_values,
            self.iterator.fm_feat_shape,
        )
        xx = tf.sparse.SparseTensor(
            self.iterator.fm_feat_indices,
            tf.pow(self.iterator.fm_feat_values, 2),
            self.iterator.fm_feat_shape,
        )
        fm_output = 0.5 * tf.reduce_sum(
            input_tensor=tf.pow(tf.sparse.sparse_dense_matmul(x, embedding), 2)
            - tf.sparse.sparse_dense_matmul(xx, tf.pow(embedding, 2)),
            axis=1,
            keepdims=True,
        )
        return fm_output

class CIN(tf.keras.layers.Layer):
    """
    Get the compressed interaction network.
    This component provides explicit and vector-wise higher-order feature interactions.
    Args:
        res (bool): Whether use residual structure to fuse the results from each layer of CIN.
        direct (bool): If true, then all hidden units are connected to both next layer and output layer;
                otherwise, half of hidden units are connected to next layer and the other half will be connected to output layer.
        bias (bool): Whether to add bias term when calculating the feature maps.
        is_masked (bool): Controls whether to remove self-interaction in the first layer of CIN.
    Returns:
        object: Prediction score made by CIN.
    """
    def __init__(self, hparams, initializer, cross_params, layer_params,
                 res=True, direct=False, bias=False, is_masked=True, **kwargs):
        super(CIN, self).__init__(**kwargs)
        self.hparams = hparams
        self.res = res      # res=True, direct=False, bias=False, is_masked=True
        self.direct = direct
        self.bias = bias
        self.is_masked = is_masked

        self.filters = []
        self.b = []
        self.BN = []

        field_nums = [int(hparams.FIELD_COUNT)]
        final_len = 0
        for idx, layer_size in enumerate(hparams.cross_layer_sizes):
            filters = tf.Variable(
                name="f_" + str(idx),
                initial_value=initializer(shape=[1, field_nums[-1] * field_nums[0], layer_size], dtype=tf.float32)
            )
            self.filters.append(filters)
            if bias:
                b = tf.Variable(
                    name="f_b" + str(idx),
                    initial_value=tf.zeros_initializer()(shape=[layer_size], dtype=tf.float32)
                )
                self.b.append(b)
                cross_params.append(b)
            if hparams.enable_BN is True:
                self.BN.append(tf.keras.layers.BatchNormalization())
            if self.direct:
                final_len += layer_size
                field_nums.append(int(layer_size))
            else:
                if idx != len(hparams.cross_layer_sizes) - 1:
                    final_len += int(layer_size / 2)
                else:
                    final_len += layer_size
                field_nums.append(int(layer_size / 2))

            cross_params.append(filters)

        self.w_nn_output = tf.Variable(
            name="w_nn_output", initial_value=initializer(shape=[final_len, 1], dtype=tf.float32)
        )
        self.b_nn_output = tf.Variable(
            name="b_nn_output", initial_value=tf.zeros_initializer()(shape=[1], dtype=tf.float32)
        )
        layer_params.append(self.w_nn_output)
        layer_params.append(self.b_nn_output)

    def call(self, nn_input):
        # nn_input (object): The output of field-embedding layer. This is the input for CIN.
        hparams = self.hparams
        nn_input = tf.reshape(nn_input, shape=[-1, int(hparams.FIELD_COUNT), hparams.dim])

        final_len = 0
        hidden_nn_layers = [nn_input]
        field_nums = [int(hparams.FIELD_COUNT)]

        final_result = []
        split_tensor0 = tf.split(hidden_nn_layers[0], hparams.dim * [1], 2)

        for idx, layer_size in enumerate(hparams.cross_layer_sizes):
            split_tensor = tf.split(hidden_nn_layers[-1], hparams.dim * [1], 2)
            dot_result_m = tf.matmul(
                split_tensor0, split_tensor, transpose_b=True
            )  # shape :  (Dim, Batch, FieldNum, HiddenNum), a.k.a (D,B,F,H)
            dot_result_o = tf.reshape(
                dot_result_m,
                shape=[hparams.dim, -1, field_nums[0] * field_nums[-1]],
            )  # shape: (D,B,FH)
            dot_result = tf.transpose(a=dot_result_o, perm=[1, 0, 2])  # (B,D,FH)

            filters = self.filters[idx]

            if self.is_masked and idx == 0:
                ones = tf.ones([field_nums[0], field_nums[0]], dtype=tf.float32)
                mask_matrix = tf.linalg.band_part(
                    ones, 0, -1
                ) - tf.linalg.tensor_diag(tf.ones(field_nums[0])) # upper part (not even the diagonal)
                mask_matrix = tf.reshape(
                    mask_matrix, shape=[1, field_nums[0] * field_nums[0]]
                )

                dot_result = tf.multiply(dot_result, mask_matrix) * 2
                # self.dot_result = dot_result

            curr_out = tf.nn.conv1d(
                input=dot_result, filters=filters, stride=1, padding="VALID"
            )  # shape : (B,D,H`)

            if self.bias:
                b = self.b[idx]
                curr_out = tf.nn.bias_add(curr_out, b)

            if hparams.enable_BN is True:
                curr_out = self.BN[idx](curr_out, training=self.trainable)

            curr_out = _activate(curr_out, hparams.cross_activation)
            curr_out = tf.transpose(a=curr_out, perm=[0, 2, 1])  # shape : (B,H,D)

            if self.direct:
                direct_connect = curr_out
                next_hidden = curr_out
                final_len += layer_size
                field_nums.append(int(layer_size))
            else:
                if idx != len(hparams.cross_layer_sizes) - 1:
                    next_hidden, direct_connect = tf.split(
                        curr_out, 2 * [int(layer_size / 2)], 1
                    )
                    final_len += int(layer_size / 2)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
                    final_len += layer_size
                field_nums.append(int(layer_size / 2))

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        result = tf.concat(final_result, axis=1)
        result = tf.reduce_sum(input_tensor=result, axis=-1)  # shape : (B,H)
        if self.res:
            base_score = tf.reduce_sum(
                input_tensor=result, axis=1, keepdims=True
            )  # (B,1)
        else:
            base_score = 0

        exFM_out = base_score + tf.add(tf.linalg.matmul(
            result, self.w_nn_output), self.b_nn_output
        )
        return exFM_out

class DNN(tf.keras.layers.Layer):
    """
    The MLP part for the model.
    This components provides implicit higher-order feature interactions.

    Args:
        embed_out (object): The output of field-embedding layer. This is the input for DNN.
        embed_layer_size (object): Shape of the embed_out

    Returns:
        object: Prediction score made by fast CIN.
    """
    def __init__(self, hparams, initializer, _embedding, layer_params, **kwargs):
        super(DNN, self).__init__(**kwargs)
        self.hparams = hparams
        self.initializer = initializer
        self._embedding = _embedding
        self.layer_params = layer_params

    def build(self, input_shape):
        hparams = self.hparams
        self.embed_layer_size = self._embedding.embed_layer_size

        last_layer_size = self.embed_layer_size
        layer_idx = 0
        self.w_nn_layers = []
        self.b_nn_layers = []
        self.BN = []

        for idx, layer_size in enumerate(hparams.layer_sizes):
            curr_w_nn_layer = tf.Variable(
                name="w_nn_layer" + str(layer_idx),
                initial_value=self.initializer(shape=[last_layer_size, layer_size], dtype=tf.float32)
            )
            self.w_nn_layers.append(curr_w_nn_layer)
            curr_b_nn_layer = tf.Variable(
                name="b_nn_layer" + str(layer_idx),
                initial_value=tf.zeros_initializer()(shape=[layer_size], dtype=tf.float32)
            )
            self.b_nn_layers.append(curr_b_nn_layer)

            layer_idx += 1
            last_layer_size = layer_size
            self.layer_params.append(curr_w_nn_layer)
            self.layer_params.append(curr_b_nn_layer)
            self.BN.append(tf.keras.layers.BatchNormalization())

        self.w_nn_output = tf.Variable(
            name="w_nn_output", initial_value=self.initializer(shape=[last_layer_size, 1], dtype=tf.float32)
        )
        self.b_nn_output = tf.Variable(
            name="b_nn_output", initial_value=tf.zeros_initializer()(shape=[1], dtype=tf.float32)
        )
        self.layer_params.append(self.w_nn_output)
        self.layer_params.append(self.b_nn_output)

    def call(self, embed_out):
        hparams = self.hparams
        w_fm_nn_input = embed_out

        last_layer_size = self.embed_layer_size
        layer_idx = 0
        hidden_nn_layers = []
        hidden_nn_layers.append(w_fm_nn_input)

        for idx, layer_size in enumerate(hparams.layer_sizes):
            curr_w_nn_layer = self.w_nn_layers[idx]
            curr_b_nn_layer = self.b_nn_layers[idx]
            tf.summary.histogram(
                "nn_part/" + "w_nn_layer" + str(layer_idx), curr_w_nn_layer
            )
            tf.summary.histogram(
                "nn_part/" + "b_nn_layer" + str(layer_idx), curr_b_nn_layer
            )
            curr_hidden_nn_layer = tf.add(tf.linalg.matmul(
                hidden_nn_layers[layer_idx], curr_w_nn_layer), curr_b_nn_layer
            )
            activation = hparams.activation[idx]  # ['relu', 'relu']

            if hparams.enable_BN is True:
                curr_hidden_nn_layer = self.BN[idx](curr_hidden_nn_layer, training=self.trainable)

            curr_hidden_nn_layer = _active_layer(
                logit=curr_hidden_nn_layer, activation=activation,
                layer_keeps=self.layer_keeps, layer_idx=idx, user_dropout=hparams.user_dropout and self.trainable
            )

            hidden_nn_layers.append(curr_hidden_nn_layer)
            layer_idx += 1
            last_layer_size = layer_size

        w_nn_output = self.w_nn_output
        b_nn_output = self.b_nn_output
        tf.summary.histogram(
            "nn_part/" + "w_nn_output" + str(layer_idx), w_nn_output
        )
        tf.summary.histogram(
            "nn_part/" + "b_nn_output" + str(layer_idx), b_nn_output
        )

        nn_output = tf.add(tf.linalg.matmul(
            hidden_nn_layers[-1], w_nn_output), b_nn_output
        )
        return nn_output

class XDeepFMModel_v2(BaseModel_v2):

    def _build_model(self):
        hparams = self.hparams
        self._embedding = Embedding(hparams, self.initializer, self.iterator, self.embed_params)

        if hparams.use_Linear_part:
            self._linear = Linear(hparams, self.initializer, self.iterator, self.layer_params)

        if hparams.use_FM_part:
            self._fm = FM(self._embedding, self.iterator)

        if hparams.use_DNN_part:
            self._dnn = DNN(hparams, self.initializer, self._embedding, self.layer_params)

        assert(hparams.fast_CIN_d <= 0)
        if hparams.use_CIN_part:
            self._cin = CIN(hparams, self.initializer, self.cross_params, self.layer_params)
            # res=True, direct=False, bias=False, is_masked=True

    def set_trainable(self, is_trainable):
        hparams = self.hparams
        self._embedding.trainable = is_trainable

        if hparams.use_Linear_part:
            self._linear.trainable = is_trainable
        if hparams.use_FM_part:
            self._fm.trainable = is_trainable
        if hparams.use_DNN_part:
            self._dnn.trainable = is_trainable
            if is_trainable:
                self._dnn.layer_keeps = self.keep_prob_train
            else:
                self._dnn.layer_keeps = self.keep_prob_test
        if hparams.use_CIN_part:
            self._cin.trainable = is_trainable

    def _call_model(self, feed_dict):
        """
        The main function to create xdeepfm's logic.
        Returns:
            object: The prediction score made by the model.
        """
        hparams = self.hparams
        logit = 0
        embed_out = self._embedding(feed_dict)

        if hparams.use_Linear_part:
            logit = logit + self._linear(feed_dict)

        if hparams.use_FM_part:
            logit = logit + self._fm(feed_dict)

        if hparams.use_CIN_part:
            logit = logit + self._cin(embed_out)

        if hparams.use_DNN_part:
            logit = logit + self._dnn(embed_out)

        return logit
