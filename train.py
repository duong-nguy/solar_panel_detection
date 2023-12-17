import os
import shutil
import tifffile

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.dtensor import utils as dtensor_utils
from keras.metrics import base_metric


from keras_unet_collection import models,losses

def data_augment(x,y):
    """
    Augment data for semantic segmentation.
    Args:
        image (numpy.array): image numpy array
        label (numpy.array): image numpy array
    Return:
        augmented image and label
    ----------
    Example
    ----------
        data_augment(image, label)
    """
    # Thanks to the dataset.prefetch(AUTO) statement in the next function
    # (below), this happens essentially for free on TPU. Data pipeline code
    # is executed on the CPU part of the TPU, TPU is computing gradients.
    randint = np.random.randint(1, 7)
    if randint == 1:  # flip left and right
        x = tf.image.random_flip_left_right(x)
        y = tf.image.random_flip_left_right(y)
    elif randint == 2:  # reverse second dimension
        x = tf.image.random_flip_up_down(x)
        y = tf.image.random_flip_up_down(y)
    elif randint == 3:  # rotate 90 degrees
        x = tf.image.rot90(x, k=1)
        y = tf.image.rot90(y, k=1)
    elif randint == 4:  # rotate 180 degrees
        x = tf.image.rot90(x, k=2)
        y = tf.image.rot90(y, k=2)
    elif randint == 5:  # rotate 270 degrees
        x = tf.image.rot90(x, k=3)
        y = tf.image.rot90(y, k=3)
    return x, y

def unet(input_shape,output_shape,normalize=False):
    def unet_conv(inputs,fillters,down,dropout=None,normalize=False,kernel_regularizer=None):
        skip_conect = None
        x = tf.keras.layers.Conv2D(fillters, 3,
                                   padding='same',
                                   kernel_initializer='he_normal',
                                   use_bias=not normalize,
                                   kernel_regularizer=kernel_regularizer
                                   )(inputs)
        if normalize:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(fillters, 3,
                                   padding='same',
                                   kernel_initializer='he_normal',
                                   use_bias=not normalize,
                                   kernel_regularizer=kernel_regularizer
                                   )(x)
        if normalize:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        skip_conect = x

        if dropout != None:
            x = tf.keras.layers.Dropout(dropout)(x)
            skip_conect = x

        if down:
            if x.shape[1] != 1:
                x = tf.keras.layers.MaxPooling2D(2)(x)
        else:
            x = tf.keras.layers.UpSampling2D(size=(2,2),interpolation='bilinear')(x)
        return x,skip_conect
    inputs = tf.keras.Input(input_shape)
    ########
    N = 32
    ########
    #UNET encoder (down sample)
    x,skip_conect1 = unet_conv(inputs,N*1,down=True,
                           normalize=normalize,)#13
    x,skip_conect2 = unet_conv(x,N*2,down=True,
                           normalize=normalize)#6
    x,skip_conect3 = unet_conv(x,N*3,down=True,
                           normalize=normalize)#3
    x,skip_conect4 = unet_conv(x,N*4,down=True,
                               dropout=None,
                               normalize=normalize,)
#                                kernel_regularizer=tf.keras.regularizers.L1(l1=0.01))#2
    x,skip_conect5 = unet_conv(x,N*5,down=True,
                               dropout=None,
                               normalize=normalize,)
#                                kernel_regularizer=tf.keras.regularizers.L1(l1=0.01))#1
    #UNET decoder (up sample)
    x,_ = unet_conv(x,N*4,down=False,
                         normalize=normalize,)
#                          kernel_regularizer=tf.keras.regularizers.L1(l1=0.01))
    x = tf.keras.layers.ZeroPadding2D(
            padding=((1,0),(1,0)))(x)
    x = tf.keras.layers.Concatenate()([x,skip_conect4]) #3,3

    x,_ = unet_conv(x,N*3,down=False,
                         normalize=normalize)
    x = tf.keras.layers.Concatenate()([x,skip_conect3])#6,6

    x,_ = unet_conv(x,N*2,down=False,
                         normalize=normalize)#12
    x = tf.keras.layers.ZeroPadding2D(
            padding=((1,0),(1,0)))(x)#13
    x = tf.keras.layers.Concatenate()([x,skip_conect2])#13,13

    x,_ = unet_conv(x,N*1,down=False,
                         normalize=normalize)
    x = tf.keras.layers.Concatenate()([x,skip_conect1])#26,26

    outputs = tf.keras.layers.Conv2D(output_shape[-1],1,activation='sigmoid')(x)

    model = tf.keras.Model(inputs,outputs)

    return model

def unet_plus_2d(input_shape,output_shape,normalize=False):
    model = models.unet_plus_2d(input_shape, [64, 128, 256, 512, 1024, 2028], n_labels=output_shape[-1],
                            stack_num_down=2, stack_num_up=2,
                            activation='ReLU', output_activation='Sigmoid',
                            batch_norm=normalize, pool='max', unpool=False, deep_supervision=True, name='unnet_plus_2d')
    return model
    

def debug_model(input_shape,output_shape):
    inputs = tf.keras.Input(input_shape)
    outputs = tf.keras.layers.Conv2D(1,1,activation='sigmoid')(inputs)
    model = tf.keras.Model(inputs,outputs)
    return model


def hybrid_loss(y_true, y_pred):

    loss_focal = losses.focal_tversky(y_true, y_pred, alpha=0.5, gamma=4/3)
    loss_iou = losses.iou_seg(y_true, y_pred)

    # (x)
    #loss_ssim = losses.ms_ssim(y_true, y_pred, max_val=1.0, filter_size=4)

    return loss_focal+loss_iou

class Train:
    def __init__(self,
                 project_root,
                 model,
                 loss,
                 optimizer,
                 batch,
                 epochs,
                 metrics=None,
                 callbacks=None,
                 ):
        self.root = project_root
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.callbacks = callbacks
        self.optimizer = optimizer
        self.batch = batch
        self.epochs = epochs
        self.outputs_path = os.path.join(self.root,'train')
        self.train_dataset_path = os.path.join(self.root,'feature_selection/train_dataset')

        if not os.path.exists(self.root):
            raise FileNotFoundError(f'Project {root} is not exists. Please run Preprocessor first')

        if not os.path.exists(self.outputs_path):
            print(f'Create {self.outputs_path}')
            os.mkdir(self.outputs_path)

        else:
            print(f'Remove old {self.outputs_path}')
            shutil.rmtree(self.outputs_path)
            print(f'Create new {self.outputs_path}')
            os.mkdir(self.outputs_path)

    def _get_dataset(self):
        dataset = tf.data.Dataset.load(self.train_dataset_path,compression='GZIP')
        dataset_len = len(dataset)
        num_train = (dataset_len // 10) * 8

        train_dataset = dataset.take(num_train)
        valid_dataset = dataset.skip(num_train)

        train_dataset = train_dataset.batch(self.batch).cache().map(data_augment).prefetch(
                tf.data.AUTOTUNE)
        valid_dataset = valid_dataset.batch(self.batch).cache().prefetch(
                tf.data.AUTOTUNE)
        return train_dataset,valid_dataset
    
    def _get_model(self,input_shape,output_shape):
        model = self.model(input_shape,output_shape)
        model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics
            )
        return model

    def _get_metadata(self,data_sample):
        image,mask = next(iter(data_sample.unbatch()))
        input_shape = image.shape
        output_shape = mask.shape
        return input_shape,output_shape

    def _train(self,train_dataset,valid_dataset):
        input_shape,output_shape = self._get_metadata(train_dataset.take(1))
        model = self._get_model(input_shape,output_shape)
        history = model.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=self.epochs,
            callbacks=self.callbacks,
            verbose=2
        )

        model_save_name = os.path.join(self.outputs_path,f'{self.root}_model.keras')
        model.save(model_save_name)
#        filename = os.path.join(self.outputs_path,'history.txt')
#        with open(filename, 'w') as file:
#            file.write(str(history.history))

    def run(self):
        train_dataset,valid_dataset = self._get_dataset()
        self._train(train_dataset,valid_dataset)

class FBetaScore(base_metric.Metric):
    """Computes F-Beta score.

    THIS FUNCTION IS MORDIFITED TO WORK WITH 2D (IMAGE) DATA
    THERE MAYBE SOME BUGS

    This is the weighted harmonic mean of precision and recall.
    Its output range is `[0, 1]`. It works for both multi-class
    and multi-label classification.

    It is defined as:

    ```python

    b2 = beta ** 2
    f_beta_score = (1 + b2) * (precision * recall) / (precision * b2 + recall)
    ```

    Args:
        average: Type of averaging to be performed across per-class results
            in the multi-class case.
            Acceptable values are `None`, `"micro"`, `"macro"` and
            `"weighted"`. Default value is `None`.
            If `None`, no averaging is performed and `result()` will return
            the score for each class.
            If `"micro"`, compute metrics globally by counting the total
            true positives, false negatives and false positives.
            If `"macro"`, compute metrics for each label,
            and return their unweighted mean.
            This does not take label imbalance into account.
            If `"weighted"`, compute metrics for each label,
            and return their average weighted by support
            (the number of true instances for each label).
            This alters `"macro"` to account for label imbalance.
            It can result in an score that is not between precision and recall.
        beta: Determines the weight of given to recall
            in the harmonic mean between precision and recall (see pseudocode
            equation above). Default value is 1.
        threshold: Elements of `y_pred` greater than `threshold` are
            converted to be 1, and the rest 0. If `threshold` is
            `None`, the argmax of `y_pred` is converted to 1, and the rest to 0.
        name: Optional. String name of the metric instance.
        dtype: Optional. Data type of the metric result.

    Returns:
        F-Beta Score: float.

    Example:

    >>> metric = tf.keras.metrics.FBetaScore(beta=2.0, threshold=0.5)
    >>> y_true = np.array([[1, 1, 1],
    ...                    [1, 0, 0],
    ...                    [1, 1, 0]], np.int32)
    >>> y_pred = np.array([[0.2, 0.6, 0.7],
    ...                    [0.2, 0.6, 0.6],
    ...                    [0.6, 0.8, 0.0]], np.float32)
    >>> metric.update_state(y_true, y_pred)
    >>> result = metric.result()
    >>> result.numpy()
    array([0.3846154 , 0.90909094, 0.8333334 ], dtype=float32)
    """

    @dtensor_utils.inject_mesh
    def __init__(
        self,
        average=None,
        beta=1.0,
        threshold=None,
        name="fbeta_score",
        dtype=None,
    ):
        super().__init__(name=name, dtype=dtype)

        if average not in (None, "micro", "macro", "weighted"):
            raise ValueError(
                "Invalid `average` argument value. Expected one of: "
                "{None, 'micro', 'macro', 'weighted'}. "
                f"Received: average={average}"
            )

        if not isinstance(beta, float):
            raise ValueError(
                "Invalid `beta` argument value. "
                "It should be a Python float. "
                f"Received: beta={beta} of type '{type(beta)}'"
            )
        if beta <= 0.0:
            raise ValueError(
                "Invalid `beta` argument value. "
                "It should be > 0. "
                f"Received: beta={beta}"
            )

        if threshold is not None:
            if not isinstance(threshold, float):
                raise ValueError(
                    "Invalid `threshold` argument value. "
                    "It should be a Python float. "
                    f"Received: threshold={threshold} "
                    f"of type '{type(threshold)}'"
                )
            if threshold > 1.0 or threshold <= 0.0:
                raise ValueError(
                    "Invalid `threshold` argument value. "
                    "It should verify 0 < threshold <= 1. "
                    f"Received: threshold={threshold}"
                )

        self.average = average
        self.beta = beta
        self.threshold = threshold
        self.axis = None
        self.built = False

        if self.average != "micro":
            self.axis = 0

    def build(self, y_true_shape, y_pred_shape):
#         if len(y_pred_shape) != 2 or len(y_true_shape) != 2:
#             raise ValueError(
#                 "FBetaScore expects 2D inputs with shape "
#                 "(batch_size, output_dim). Received input "
#                 f"shapes: y_pred.shape={y_pred_shape} and "
#                 f"y_true.shape={y_true_shape}."
#             )
        if y_pred_shape[-1] is None or y_true_shape[-1] is None:
            raise ValueError(
                "FBetaScore expects 2D inputs with shape "
                "(batch_size, output_dim), with output_dim fully "
                "defined (not None). Received input "
                f"shapes: y_pred.shape={y_pred_shape} and "
                f"y_true.shape={y_true_shape}."
            )
        num_classes = y_pred_shape[-1]
        if self.average != "micro":
            init_shape = [num_classes]
        else:
            init_shape = []

        def _add_zeros_weight(name):
            return self.add_weight(
                name,
                shape=init_shape,
                initializer="zeros",
                dtype=self.dtype,
            )

        self.true_positives = _add_zeros_weight("true_positives")
        self.false_positives = _add_zeros_weight("false_positives")
        self.false_negatives = _add_zeros_weight("false_negatives")
        self.intermediate_weights = _add_zeros_weight("intermediate_weights")
        self.built = True

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.convert_to_tensor(y_true, dtype=self.dtype)
        y_pred = tf.convert_to_tensor(y_pred, dtype=self.dtype)
        if not self.built:
            self.build(y_true.shape, y_pred.shape)

        if self.threshold is None:
            threshold = tf.reduce_max(y_pred, axis=-1, keepdims=True)
            # make sure [0, 0, 0] doesn't become [1, 1, 1]
            # Use abs(x) > eps, instead of x != 0 to check for zero
            y_pred = tf.logical_and(y_pred >= threshold, tf.abs(y_pred) > 1e-9)
        else:
            y_pred = y_pred > self.threshold
        y_pred = tf.cast(y_pred, dtype=self.dtype)

        def _weighted_sum(val, sample_weight):
            if sample_weight is not None:
                val = tf.math.multiply(val, tf.expand_dims(sample_weight, 1))
            return tf.reduce_sum(val, axis=[0,1,2])# magic numbers

        self.true_positives.assign_add(
            _weighted_sum(y_pred * y_true, sample_weight)
        )
        self.false_positives.assign_add(
            _weighted_sum(y_pred * (1 - y_true), sample_weight)
        )
        self.false_negatives.assign_add(
            _weighted_sum((1 - y_pred) * y_true, sample_weight)
        )
        self.intermediate_weights.assign_add(
            _weighted_sum(y_true, sample_weight)
        )

    def result(self):
        precision = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_positives
        )
        recall = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_negatives
        )

        mul_value = precision * recall
        add_value = (tf.math.square(self.beta) * precision) + recall
        mean = tf.math.divide_no_nan(mul_value, add_value)
        f1_score = mean * (1 + tf.math.square(self.beta))

        if self.average == "weighted":
            weights = tf.math.divide_no_nan(
                self.intermediate_weights,
                tf.reduce_sum(self.intermediate_weights),
            )
            f1_score = tf.reduce_sum(f1_score * weights)

        elif self.average is not None:  # [micro, macro]
            f1_score = tf.reduce_mean(f1_score)

        return f1_score

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "average": self.average,
            "beta": self.beta,
            "threshold": self.threshold,
        }

        base_config = super().get_config()
        return {**base_config, **config}

    def reset_state(self):
        for v in self.variables:
            v.assign(tf.zeros(v.shape, dtype=v.dtype))



@keras.saving.register_keras_serializable(package="F1Score")
class F1Score(FBetaScore):
    r"""Computes F-1 Score.
    This is the harmonic mean of precision and recall.
    Its output range is `[0, 1]`. It works for both multi-class
    and multi-label classification.

    It is defined as:

    ```python
    f1_score = 2 * (precision * recall) / (precision + recall)
    ```

    Args:
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `"micro"`, `"macro"`
            and `"weighted"`. Default value is `None`.
            If `None`, no averaging is performed and `result()` will return
            the score for each class.
            If `"micro"`, compute metrics globally by counting the total
            true positives, false negatives and false positives.
            If `"macro"`, compute metrics for each label,
            and return their unweighted mean.
            This does not take label imbalance into account.
            If `"weighted"`, compute metrics for each label,
            and return their average weighted by support
            (the number of true instances for each label).
            This alters `"macro"` to account for label imbalance.
            It can result in an score that is not between precision and recall.
        threshold: Elements of `y_pred` greater than `threshold` are
            converted to be 1, and the rest 0. If `threshold` is
            `None`, the argmax of `y_pred` is converted to 1, and the rest to 0.
        name: Optional. String name of the metric instance.
        dtype: Optional. Data type of the metric result.

    Returns:
        F-1 Score: float.

    Example:

    >>> metric = tf.keras.metrics.F1Score(threshold=0.5)
    >>> y_true = np.array([[1, 1, 1],
    ...                    [1, 0, 0],
    ...                    [1, 1, 0]], np.int32)
    >>> y_pred = np.array([[0.2, 0.6, 0.7],
    ...                    [0.2, 0.6, 0.6],
    ...                    [0.6, 0.8, 0.0]], np.float32)
    >>> metric.update_state(y_true, y_pred)
    >>> result = metric.result()
    >>> result.numpy()
    array([0.5      , 0.8      , 0.6666667], dtype=float32)
    """

    @dtensor_utils.inject_mesh
    def __init__(
        self,
        average=None,
        threshold=None,
        name="f1_score",
        dtype=None,
    ):
        super().__init__(
            average=average,
            beta=1.0,
            threshold=threshold,
            name=name,
            dtype=dtype,
        )

    def get_config(self):
        base_config = super().get_config()
        del base_config["beta"]
        return base_config
