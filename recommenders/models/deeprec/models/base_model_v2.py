from os.path import join
import abc
import time
import os
import numpy as np
import tensorflow as tf
from recommenders.models.deeprec.deeprec_utils import cal_metric


__all__ = ["BaseModel_v2"]

def _get_initializer(init_method, init_value, seed):
    if init_method == "tnormal":
        return tf.keras.initializers.TruncatedNormal(
            stddev=init_value, seed=seed
        )
    elif init_method == "uniform":
        return tf.keras.initializers.RandomUniform(
            -init_value, init_value, seed=seed
        )
    elif init_method == "normal":
        return tf.keras.initializers.RandomNormal(
            stddev=init_value, seed=seed
        )
    elif init_method == "xavier_normal":
        return tf.keras.initializers.VarianceScaling(
            scale=1.0,
            mode="fan_avg",
            distribution=("uniform" if False else "truncated_normal"),
            seed=seed,
        )
    elif init_method == "xavier_uniform":
        return tf.keras.initializers.VarianceScaling(
            scale=1.0,
            mode="fan_avg",
            distribution=("uniform" if True else "truncated_normal"),
            seed=seed,
        )
    elif init_method == "he_normal":
        return tf.keras.initializers.VarianceScaling(
            scale=2.0,
            mode=("FAN_IN").lower(),
            distribution=("uniform" if False else "truncated_normal"),
            seed=seed,
        )
    elif init_method == "he_uniform":
        return tf.keras.initializers.VarianceScaling(
            scale=2.0,
            mode=("FAN_IN").lower(),
            distribution=("uniform" if True else "truncated_normal"),
            seed=seed,
        )
    else:
        return tf.keras.initializers.TruncatedNormal(
            stddev=init_value, seed=seed
        )

def _activate(logit, activation):
    if activation == "sigmoid":
        return tf.nn.sigmoid(logit)
    elif activation == "softmax":
        return tf.nn.softmax(logit)
    elif activation == "relu":
        return tf.nn.relu(logit)
    elif activation == "tanh":
        return tf.nn.tanh(logit)
    elif activation == "elu":
        return tf.nn.elu(logit)
    elif activation == "identity":
        return tf.identity(logit)
    else:
        raise ValueError("this activations not defined {0}".format(activation))

def _dropout(logit, keep_prob):
    """Apply drops upon the input value.
    Args:
        logit (object): The input value.
        keep_prob (float): The probability of keeping each element.
    Returns:
        object: A tensor of the same shape of logit.
    """
    return tf.nn.dropout(x=logit, rate=1 - (keep_prob))

def _train_opt(lr, optimizer):
    """Get the optimizer according to configuration. Usually we will use Adam.
    Returns:
        object: An optimizer.
    """
    if optimizer == "adadelta":
        train_step = tf.keras.optimizers.Adadelta(lr)
    elif optimizer == "adagrad":
        train_step = tf.keras.optimizers.Adagrad(lr)
    elif optimizer == "sgd":
        train_step = tf.keras.optimizers.SGD(lr)
    elif optimizer == "adam":
        train_step = tf.keras.optimizers.Adam(lr)
    elif optimizer == "ftrl":
        train_step = tf.keras.optimizers.Ftrl(lr)
    elif optimizer == "gd":
        train_step = tf.keras.optimizers.SGD(lr)
    elif optimizer == "rmsprop":
        train_step = tf.keras.optimizers.RMSprop(lr)
    else:
        train_step = tf.keras.optimizers.SGD(lr)
    return train_step

def _active_layer(logit, activation, layer_keeps, layer_idx=-1, user_dropout=False):
    """Transform the input value with an activation. May use dropout.
    Args:
        logit (object): Input value.
        activation (str): A string indicating the type of activation function.
        layer_keeps (array): A numpy array with the prob of keeping
        layer_idx (int): Index of current layer. Used to retrieve corresponding parameters
    Returns:
        object: A tensor after applying activation function on logit.
    """
    if layer_idx >= 0 and user_dropout:
        logit = _dropout(logit, layer_keeps[layer_idx])
    return _activate(logit, activation)

class BaseModel_v2:
    """Base class for models"""

    def __init__(self, hparams, iterator_creator, seed=None):
        """Initializing the model. Create common logics which are needed by all deeprec models, such as loss function,
        parameter set.
        Args:
            hparams (object): An `HParams` object, holds the entire set of hyperparameters.
            iterator_creator (object): An iterator to load the data.
            seed (int): Random seed.
        """
        self.hparams = hparams

        tf.random.set_seed(seed)
        np.random.seed(seed)

        self.iterator = iterator_creator(hparams)
        self.train_num_ngs = (
            hparams.train_num_ngs if "train_num_ngs" in hparams.values() else None
        )

        self.layer_params = []
        self.embed_params = []
        self.cross_params = []

        self.keep_prob_train = 1 - np.array(hparams.dropout)
        self.keep_prob_test = np.ones_like(hparams.dropout)

        self.initializer = _get_initializer(hparams.init_method, hparams.init_value, seed)

        self._build_model()
        self.optimizer = _train_opt(hparams.learning_rate, hparams.optimizer)
        self.trainable_params = None

        #  self.saver = tf.compat.v1.train.Saver(max_to_keep=self.hparams.epochs)

    @abc.abstractmethod
    def _build_model(self):
        """Subclass will implement this."""
        pass

    @abc.abstractmethod
    def _call_model(self, feed_dict):
        pass


    def _get_pred(self, feed_dict):
        """Make final output as prediction score, according to different tasks.
        Args:
            logit (object): Base prediction value.
            task (str): A task (values: regression/classification)
        Returns:
            object: Transformed score.
        """
        task = self.hparams.method
        logit = self._call_model(feed_dict)

        if task == "regression":
            pred = tf.identity(logit)
        elif task == "classification":
            pred = tf.keras.activations.sigmoid(logit)
        else:
            raise ValueError(
                "method must be regression or classification, but now is {0}".format(
                    task
                )
            )
        pred = tf.identity(pred, name="pred")
        return pred

    def _get_loss(self):
        """Make loss function, consists of data loss and regularization loss

        Returns:
            object: Loss value.
        """
        self.data_loss = self._compute_data_loss()
        self.regular_loss = self._compute_regular_loss()
        self.loss = tf.add(self.data_loss, self.regular_loss)
        return self.loss

    def _add_summaries(self):
        tf.summary.scalar("data_loss", self.data_loss)
        tf.summary.scalar("regular_loss", self.regular_loss)
        tf.summary.scalar("loss", self.loss)

    def _l2_loss(self):
        l2_loss = tf.zeros([1], dtype=tf.float32)
        # embedding_layer l2 loss
        for param in self.embed_params:
            l2_loss = tf.add(
                l2_loss, tf.multiply(self.hparams.embed_l2, tf.nn.l2_loss(param))
            )
        params = self.layer_params
        for param in params:
            l2_loss = tf.add(
                l2_loss, tf.multiply(self.hparams.layer_l2, tf.nn.l2_loss(param))
            )
        return l2_loss

    def _l1_loss(self):
        l1_loss = tf.zeros([1], dtype=tf.float32)
        # embedding_layer l2 loss
        for param in self.embed_params:
            l1_loss = tf.add(
                l1_loss,
                tf.multiply(self.hparams.embed_l1, tf.norm(tensor=param, ord=1)),
            )
        params = self.layer_params
        for param in params:
            l1_loss = tf.add(
                l1_loss,
                tf.multiply(self.hparams.layer_l1, tf.norm(tensor=param, ord=1)),
            )
        return l1_loss

    def _cross_l_loss(self):
        """Construct L1-norm and L2-norm on cross network parameters for loss function.

        Returns:
            object: Regular loss value on cross network parameters.
        """
        cross_l_loss = tf.zeros([1], dtype=tf.float32)
        for param in self.cross_params:
            cross_l_loss = tf.add(
                cross_l_loss,
                tf.multiply(self.hparams.cross_l1, tf.norm(tensor=param, ord=1)),
            )
            cross_l_loss = tf.add(
                cross_l_loss,
                tf.multiply(self.hparams.cross_l2, tf.norm(tensor=param, ord=2)),
            )
        return cross_l_loss

    def _compute_data_loss(self):
        if self.hparams.loss == "cross_entropy_loss":
            data_loss = tf.reduce_mean(
                input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=tf.reshape(self.logit, [-1]),
                    labels=tf.reshape(self.iterator.labels, [-1]),
                )
            )
        elif self.hparams.loss == "square_loss":
            data_loss = tf.sqrt(
                tf.reduce_mean(
                    input_tensor=tf.math.squared_difference(
                        tf.reshape(self.pred, [-1]),
                        tf.reshape(self.iterator.labels, [-1]),
                    )
                )
            )
        elif self.hparams.loss == "log_loss":
            bce = tf.keras.losses.BinaryCrossentropy()
            data_loss = bce(tf.reshape(self.iterator.labels, [-1]),
                            tf.reshape(self.pred, [-1]))
        elif self.hparams.loss == "softmax":
            group = self.train_num_ngs + 1
            logits = tf.reshape(self.logit, (-1, group))
            if self.hparams.model_type == "NextItNet":
                labels = (
                    tf.transpose(
                        a=tf.reshape(
                            self.iterator.labels,
                            (-1, group, self.hparams.max_seq_length),
                        ),
                        perm=[0, 2, 1],
                    ),
                )
                labels = tf.reshape(labels, (-1, group))
            else:
                labels = tf.reshape(self.iterator.labels, (-1, group))
            softmax_pred = tf.nn.softmax(logits, axis=-1)
            boolean_mask = tf.equal(labels, tf.ones_like(labels))
            mask_paddings = tf.ones_like(softmax_pred)
            pos_softmax = tf.where(boolean_mask, softmax_pred, mask_paddings)
            data_loss = -group * tf.reduce_mean(input_tensor=tf.math.log(pos_softmax))
        else:
            raise ValueError("this loss not defined {0}".format(self.hparams.loss))
        return data_loss

    def _compute_regular_loss(self):
        """Construct regular loss. Usually it's comprised of l1 and l2 norm.
        Users can designate which norm to be included via config file.

        Returns:
            object: Regular loss.
        """
        regular_loss = self._l2_loss() + self._l1_loss() + self._cross_l_loss()
        return tf.reduce_sum(input_tensor=regular_loss)

    @abc.abstractmethod
    def set_trainable(self, is_trainable):
        pass

    def train(self, feed_dict):
        """Go through the optimization step once with training data in `feed_dict`.

        Args:
            sess (object): The model session object.
            feed_dict (dict): Feed values to train the model. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of values, including update operation, total loss, data loss, and merged summary.
        """
        with tf.GradientTape() as tape:
            self.pred = self._get_pred(feed_dict)
            loss = self._get_loss()
        if self.trainable_params is None:
            self.trainable_params = self.embed_params + self.layer_params + self.cross_params
        gradients = tape.gradient(loss, self.trainable_params)
        if self.hparams.is_clip_norm:
            gradients = [
                None
                if gradient is None
                else tf.clip_by_norm(gradient, self.hparams.max_grad_norm)
                for gradient in gradients
            ]
        self.optimizer.apply_gradients(zip(gradients, self.trainable_params))
        return

    def eval(self, feed_dict):
        """Evaluate the data in `feed_dict` with current model.
        Args:
            feed_dict (dict): Feed values for evaluation. This is a dictionary that maps graph elements to values.
        Returns:
            list: A list of evaluated results, including total loss value, data loss value, predicted scores, and ground-truth labels.
        """
        return self._get_pred(feed_dict)

    def infer(self, feed_dict):
        """Given feature data (in `feed_dict`), get predicted scores with current model.
        Args:
            feed_dict (dict): Instances to predict. This is a dictionary that maps graph elements to values.
        Returns:
            list: Predicted scores for the given instances.
        """
        return self._get_pred(feed_dict)

    def load_model(self, model_path=None):
        """Load an existing model.
        Args:
            model_path: model path.
        Raises:
            IOError: if the restore operation failed.
        """
        pass

    def fit(self, train_file, valid_file, test_file=None):
        """
        Fit the model with `train_file`. Evaluate the model on valid_file per epoch to observe the training status.
        If `test_file` is not None, evaluate it too.
        Args:
            train_file (str): training data set.
            valid_file (str): validation set.
            test_file (str): test set.
        Returns:
            object: An instance of self.
        """
        if self.hparams.write_tfevents:
            self.writer = tf.summary.create_file_writer(
                self.hparams.SUMMARIES_DIR
            )

        for epoch in range(1, self.hparams.epochs + 1):
            step = 0
            self.hparams.current_epoch = epoch

            epoch_loss = 0
            train_start = time.time()
            self.set_trainable(True)
            for (
                batch_data_input,
                impression,
                data_size,
            ) in self.iterator.load_data_from_file(train_file):
                self.train(batch_data_input)
                self._add_summaries()
                if self.hparams.write_tfevents:
                    self.writer.add_summary(summary, step)
                epoch_loss += self.loss
                step += 1
                if step % self.hparams.show_step == 0:
                    print(
                        "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                            step, self.loss, self.data_loss
                        )
                    )

            train_end = time.time()
            train_time = train_end - train_start

            # if self.hparams.save_model:
            #     if not os.path.exists(self.hparams.MODEL_DIR):
            #         os.makedirs(self.hparams.MODEL_DIR)
            #     if epoch % self.hparams.save_epoch == 0:
            #         save_path_str = join(self.hparams.MODEL_DIR, "epoch_" + str(epoch))
            #         self.saver.save(sess=train_sess, save_path=save_path_str)

            # self.set_trainable(False) # done in run_eval
            eval_start = time.time()
            eval_res = self.run_eval(valid_file)
            train_info = ",".join(
                [
                    str(item[0]) + ":" + str(item[1])
                    for item in [("logloss loss", (epoch_loss / step).numpy())]
                ]
            )
            eval_info = ", ".join(
                [
                    str(item[0]) + ":" + str(item[1])
                    for item in sorted(eval_res.items(), key=lambda x: x[0])
                ]
            )
            if test_file is not None:
                test_res = self.run_eval(test_file)
                test_info = ", ".join(
                    [
                        str(item[0]) + ":" + str(item[1])
                        for item in sorted(test_res.items(), key=lambda x: x[0])
                    ]
                )
            eval_end = time.time()
            eval_time = eval_end - eval_start

            if test_file is not None:
                print(
                    "at epoch {0:d}".format(epoch)
                    + "\ntrain info: "
                    + train_info
                    + "\neval info: "
                    + eval_info
                    + "\ntest info: "
                    + test_info
                )
            else:
                print(
                    "at epoch {0:d}".format(epoch)
                    + "\ntrain info: "
                    + train_info
                    + "\neval info: "
                    + eval_info
                )
            print(
                "at epoch {0:d} , train time: {1:.1f} eval time: {2:.1f}".format(
                    epoch, train_time, eval_time
                )
            )

        if self.hparams.write_tfevents:
            self.writer.close()

        return self

    def group_labels(self, labels, preds, group_keys):
        """Devide `labels` and `preds` into several group according to values in group keys.

        Args:
            labels (list): ground truth label list.
            preds (list): prediction score list.
            group_keys (list): group key list.

        Returns:
            list, list:
            - Labels after group.
            - Predictions after group.
        """
        all_keys = list(set(group_keys))
        group_labels = {k: [] for k in all_keys}
        group_preds = {k: [] for k in all_keys}
        for label, p, k in zip(labels, preds, group_keys):
            group_labels[k].append(label)
            group_preds[k].append(p)
        all_labels = []
        all_preds = []
        for k in all_keys:
            all_labels.append(group_labels[k])
            all_preds.append(group_preds[k])
        return all_labels, all_preds

    def run_eval(self, filename):
        """Evaluate the given file and returns some evaluation metrics.

        Args:
            filename (str): A file name that will be evaluated.

        Returns:
            dict: A dictionary that contains evaluation metrics.
        """
        preds = []
        labels = []
        imp_indexs = []
        self.set_trainable(False)
        for batch_data_input, imp_index, data_size in self.iterator.load_data_from_file(
            filename
        ):
            step_pred = self.eval(batch_data_input)
            preds.extend(np.reshape(step_pred, -1))
            labels.extend(np.reshape(batch_data_input['labels'], -1))
            imp_indexs.extend(np.reshape(imp_index, -1))
        res = cal_metric(labels, preds, self.hparams.metrics)
        if "pairwise_metrics" in self.hparams.values():
            group_labels, group_preds = self.group_labels(labels, preds, imp_indexs)
            res_pairwise = cal_metric(
                group_labels, group_preds, self.hparams.pairwise_metrics
            )
            res.update(res_pairwise)
        return res

    def predict(self, infile_name, outfile_name):
        """Make predictions on the given data, and output predicted scores to a file.

        Args:
            infile_name (str): Input file name, format is same as train/val/test file.
            outfile_name (str): Output file name, each line is the predict score.

        Returns:
            object: An instance of self.
        """
        self.set_trainable(False)
        with tf.io.gfile.GFile(outfile_name, "w") as wt:
            for batch_data_input, _, data_size in self.iterator.load_data_from_file(
                infile_name
            ):
                step_pred = self.infer(batch_data_input)
                step_pred = step_pred[0][:data_size]
                step_pred = np.reshape(step_pred, -1)
                wt.write("\n".join(map(str, step_pred)))
                # line break after each batch.
                wt.write("\n")
        return self