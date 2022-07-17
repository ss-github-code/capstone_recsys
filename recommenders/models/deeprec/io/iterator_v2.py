
import numpy as np
import tensorflow as tf
import abc

class BaseIterator_v2(object):
    """Abstract base iterator class"""

    @abc.abstractmethod
    def parser_one_line(self, line):
        """Abstract method. Parse one string line into feature values.

        Args:
            line (str): A string indicating one instance.
        """
        pass

    @abc.abstractmethod
    def load_data_from_file(self, infile):
        """Abstract method. Read and parse data from a file.

        Args:
            infile (str): Text input file. Each line in this file is an instance.
        """
        pass

    @abc.abstractmethod
    def _convert_data(self, labels, features):
        pass

class FFMTextIterator_v2(BaseIterator_v2):
    """
    Data loader for FFM format based models, such as xDeepFM.
    Iterator will not load the whole data into memory. Instead, it loads data into memory
    per mini-batch, so that large files can be used as input data.
    """

    def __init__(self, hparams, col_spliter=" ", id_spliter="%"):
        """
        Initialize an iterator. Create the necessary placeholders for the model.
        Args:
            hparams (object): Global hyper-parameters. Some key settings such as #_feature and #_field are there.
            col_spliter (str): column splitter in one line.
            id_spliter (str): ID splitter in one line.
        """
        self.feature_cnt = hparams.FEATURE_COUNT
        self.field_cnt = hparams.FIELD_COUNT
        self.batch_size = hparams.batch_size
        self.col_spliter = col_spliter
        self.id_spliter = id_spliter

        self.labels = None
        self.fm_feat_indices = None
        self.fm_feat_values = None
        self.fm_feat_shape = None

        self.dnn_feat_indices = None
        self.dnn_feat_values = None
        self.dnn_feat_weights = None
        self.dnn_feat_shape = None

    def parser_one_line(self, line):
        """Parse one string line into feature values.
        Args:
            line (str): A string indicating one instance.
        Returns:
            list: Parsed results, including `label`, `features` and `impression_id`.
        """
        impression_id = 0
        words = line.strip().split(self.id_spliter)
        if len(words) == 2:
            impression_id = words[1].strip()
            assert False

        cols = words[0].strip().split(self.col_spliter)

        label = float(cols[0])

        features = []
        for word in cols[1:]:
            if not word.strip():
                continue
            tokens = word.split(":")
            features.append([int(tokens[0]) - 1, int(tokens[1]) - 1, float(tokens[2])])

        return label, features, impression_id

    def load_data_from_file(self, infile):
        """Read and parse data from a file.
        Args:
            infile (str): Text input file. Each line in this file is an instance.
        Returns:
            object: An iterator that yields parsed results, in the format of graph `feed_dict`.
        """
        label_list = []
        features_list = []
        impression_id_list = []
        cnt = 0

        with tf.io.gfile.GFile(infile, "r") as rd:
            for line in rd:
                label, features, impression_id = self.parser_one_line(line)

                features_list.append(features)
                label_list.append(label)
                impression_id_list.append(impression_id)

                cnt += 1
                if cnt == self.batch_size:
                    res = self._convert_data(label_list, features_list)
                    yield res, impression_id_list, self.batch_size
                    label_list = []
                    features_list = []
                    impression_id_list = []
                    cnt = 0
            if cnt > 0:
                res = self._convert_data(label_list, features_list)
                yield res, impression_id_list, cnt

    def _convert_data(self, labels, features):
        """
        Convert data into numpy arrays that are good for further operation.
        Args:
            labels (list): a list of ground-truth labels.
            features (list): a 3-dimensional list, carrying a list (batch_size) of feature array,
                    where each feature array is a list of `[field_idx, feature_idx, feature_value]` tuple.
        Returns:
            dict: A dictionary, containing tensors that are convenient for further operation.
        """
        instance_cnt = len(labels)

        fm_feat_indices = []
        fm_feat_values = []
        fm_feat_shape = [instance_cnt, self.feature_cnt]  # dense shape for FM: batch_size x feature_count

        dnn_feat_indices = []
        dnn_feat_values = []
        dnn_feat_weights = []
        dnn_feat_shape = [instance_cnt * self.field_cnt, -1]  # dense shape for DNN: batch_size*field_cnt x 1

        for i in range(instance_cnt):
            m = len(features[i])
            dnn_feat_dic = {}
            for j in range(m):
                fm_feat_indices.append([i, features[i][j][1]])
                fm_feat_values.append(features[i][j][2])
                if features[i][j][0] not in dnn_feat_dic:
                    dnn_feat_dic[features[i][j][0]] = 0
                else:
                    dnn_feat_dic[features[i][j][0]] += 1
                dnn_feat_indices.append(
                    [
                        i * self.field_cnt + features[i][j][0],
                        dnn_feat_dic[features[i][j][0]],
                    ]
                )
                dnn_feat_values.append(features[i][j][1])
                dnn_feat_weights.append(features[i][j][2])
                if dnn_feat_shape[1] < dnn_feat_dic[features[i][j][0]]:
                    dnn_feat_shape[1] = dnn_feat_dic[features[i][j][0]]
        dnn_feat_shape[1] += 1

        sorted_index = sorted(
            range(len(dnn_feat_indices)),
            key=lambda k: (dnn_feat_indices[k][0], dnn_feat_indices[k][1]),
        )

        res = {}
        self.fm_feat_indices = res["fm_feat_indices"] = tf.convert_to_tensor(np.asarray(fm_feat_indices, dtype=np.int64))
        self.fm_feat_values = res["fm_feat_values"] = tf.convert_to_tensor(np.asarray(fm_feat_values, dtype=np.float32))
        self.fm_feat_shape = res["fm_feat_shape"] = tf.convert_to_tensor(np.asarray(fm_feat_shape, dtype=np.int64))

        self.labels = res["labels"] = tf.convert_to_tensor(np.asarray([[label] for label in labels], dtype=np.float32))

        self.dnn_feat_indices = res["dnn_feat_indices"] = tf.convert_to_tensor(np.asarray(dnn_feat_indices, dtype=np.int64)[
            sorted_index
        ])
        self.dnn_feat_values = res["dnn_feat_values"] = tf.convert_to_tensor(np.asarray(dnn_feat_values, dtype=np.int64)[
            sorted_index
        ])
        self.dnn_feat_weights = res["dnn_feat_weights"] = tf.convert_to_tensor(np.asarray(dnn_feat_weights, dtype=np.float32)[
            sorted_index
        ])
        self.dnn_feat_shape = res["dnn_feat_shape"] = tf.convert_to_tensor(np.asarray(dnn_feat_shape, dtype=np.int64))
        return res
