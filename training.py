"""
This module contains utility functions and methods for training models.
"""
import os
import logging
import timeit

from tensorflow.keras import optimizers, callbacks

from ops import logits_loss, accuracy, x_corr_loss, x_chan_mean_loss
from sampling import generate_train_z_multi_output
from utils import log_and_time, get_model_out_file, get_n_t_out_directory


def train_iteration_model(model, out_dir,
                          lr=1e-3, batch_size=24, n_epochs=int(1e3), clipnorm=0.5,
                          z_bits=None, compile_kwargs=None, callback_kwargs=None):
    """
    Function for training Z-encoding models, which take Z vectors as inputs and attempt to reconstruct them as outputs.
      Lots of bells and whistles e.g. automatic early stopping, tensorboard logging, etc.

    Args:
        model: Model which expects minibatches of Z vectors as inputs and which outputs Z as well.
        out_dir: directory to store tensorboard logs in
        lr: base learning rate.
        batch_size: it's right there in the name of the input.
        n_epochs: maximum number of epochs to train for, if early stopping not triggered.
        z_bits: bits of entropy in z vectors to use for training.
        compile_kwargs: kwargs to pass to get_model_compilation_dicts function
        clipnorm: clipnorm value to pass to optimizer.
        callback_kwargs: list of callbacks to use during training.

    Returns:
        model: trained model.
        history: keras training history object.

    """
    if compile_kwargs is None:
        compile_kwargs = {}

    if callback_kwargs is None:
        callback_kwargs = {}

    callbacks = get_standard_training_callbacks(out_dir, lr, **callback_kwargs)
    # TODO - remove later. Temporary to catch diverging losses
    callbacks.append(LossThresholdStoppingCallback(-1000))

    loss_dict, loss_weight_dict, metric_dict = get_model_compilation_dicts(model.output_names, **compile_kwargs)

    optimizer = optimizers.Adam(learning_rate=lr, epsilon=1e-2, beta_1=0.95, beta_2=0.999, amsgrad=False, clipnorm=clipnorm)

    t_timer = log_and_time("Compiling model...")
    model.compile(loss=loss_dict,  optimizer=optimizer, metrics=metric_dict, loss_weights=loss_weight_dict,
                  experimental_run_tf_function=False)
    log_and_time(t_timer)

    t_timer = log_and_time("Training model...")
    data_gen = generate_train_z_multi_output(model.input.shape[1::], batch_size, len(model.outputs), entropy=z_bits)
    history = model.fit(x=data_gen, steps_per_epoch=8, epochs=n_epochs, callbacks=callbacks)
    log_and_time(t_timer)

    return model, history


def train_model_at_each_t(model_getter, model_saver, model_trainer, n_t_iterable, out_dir_base, **kwargs):
    """
    This function trains a model at each n_t produced by the input iterable and saves the results.

    If there are already trained models present in the specified output directory, those n_t values will be skipped.

    Args:
        model_getter: function which takes in n_t and returns a model
        model_trainer: function which takes in model and output directory and trains the model
        model_saver: function which takes in model and output directory and saves results
        n_t_iterable: an iterable that produces the sequences of n_t values to train models at
        out_dir_base: a base director, each trained model will be saved in a sub-directory.
        **kwargs: any additional arguments to pass to training function

    Returns:

    """
    model = None
    for n_t in n_t_iterable:

        out_dir = get_n_t_out_directory(out_dir_base, n_t)
        if os.path.exists(get_model_out_file(out_dir)):
            logging.info("***=- Found existing model in {}, skipping this n_t. -=***".format(out_dir))
        else:
            logging.info("Working on Nt={} and saving output to {}".format(str(n_t), out_dir))

            model = model_getter(model, n_t)
            model_trainer(model, out_dir, **kwargs)
            model_saver(model, out_dir)


def step_lr_schedule(epoch, lr, trigger_epochs, factor=.1):

    if epoch in trigger_epochs:
        lr = lr * factor
        logging.info("Learning rate schedule triggered, reducing LR to {}".format(lr))
    return lr


def get_model_compilation_dicts(model_output_names,
                                encoder_loss=logits_loss, encoder_metrics=accuracy,
                                x_loss=x_chan_mean_loss, x_metrics=x_chan_mean_loss,
                                loss_weights=None):
    """
    Handles creation of dictionaries specifying losses and metrics for (possibly multi-output) models.
    Args:
        model_output_names: list of output names as in Model.output_names
        encoder_loss: loss to map to any encoder outputs
        encoder_metrics: metric(s) for any encoder outputs
        x_loss: loss for X (state space) outputs
        x_metrics: metrics for X(state space) outputs
        loss_weights: A dictionary specifying weights to apply if corresponding outputs present in model.
            Setting a loss weight = 0 disables that loss.

    Returns:
        loss_dict: dict to be passed to Model.compile()
        metric_dict: dict to be passed to Model.compile()

    """
    loss_dict = {}
    metric_dict = {}
    loss_weight_dict = {}  # We recreate so only the appropriate weights are included
    for output_name in model_output_names:
        if 'encoder' in output_name:
            loss_dict[output_name] = encoder_loss
            metric_dict[output_name] = encoder_metrics
        elif 'x_output' in output_name:
            loss_dict[output_name] = x_loss
            metric_dict[output_name] = x_metrics

        if output_name in loss_weights:
            if loss_weights[output_name] == 0:
                loss_dict.pop(output_name)
            else:
                loss_weight_dict[output_name] = loss_weights[output_name]

    return loss_dict, loss_weight_dict, metric_dict


def get_standard_training_callbacks(out_dir, lr, lr_schedule=None, include_accuracy_saturation=True,
                                    reduce_lr_patience=1500, early_stopping_patience=4500):
    """
    get a bunch of callbacks, altering the defaults and allowing user-configuration of a certain sub-set of parameters
    Args:
        out_dir: output directory for tensorboard logs & model checkpoints
        lr: learning rate.
        lr_schedule: (Optional) learning rate schedule function.
        include_accuracy_saturation: If true, add the AccuracyTerminationCriteriaCallback
        reduce_lr_patience: patience value to pass to ReduceLROnPlateau
        early_stopping_patience: pateince value to pass to EarlyStopping

    Returns:
        callback_list: a list of Keras Callback objects
    """

    cb = [callbacks.TensorBoard(log_dir=out_dir, write_graph=False, profile_batch=0),
          callbacks.ReduceLROnPlateau(monitor='loss', factor=.5, patience=reduce_lr_patience, verbose=1, min_delta=1e-5,
                                      cooldown=50, min_lr=lr*.01),
          callbacks.EarlyStopping(monitor='loss', min_delta=1e-5, patience=early_stopping_patience, verbose=1),
          TrainingTimerCallback(),
          callbacks.ModelCheckpoint(get_model_out_file(out_dir), monitor='loss', verbose=1,
                                    save_freq=250*8, save_best_only=True)]
    if lr_schedule is not None:
        cb = cb + [callbacks.LearningRateScheduler(lr_schedule)]

    if include_accuracy_saturation:
        cb = cb + [AccuracyTerminationCriteriaCallback(verbose=1)]

    return cb


class AccuracyTerminationCriteriaCallback(callbacks.Callback):
    """
    A callback for terminating optimization when accuracy is saturated. Will terminate training when all of the
    specified accuracies have been at 1.0 for the specified number of epochs.
    """

    def __init__(self, patience=60, accuracy_metrics=None, verbose=0):
        """
        Args:
            patience: how long accuracc(ies) must be saturated before terminating.
            accuracy_metrics: Optional. The list of metrics to monitor. If not specified, all metrics with "accuracy"
            in the name will be monitored.
            verbose: If 1, log when terminating.
        """

        self.patience = patience
        self.accuracy_metrics = accuracy_metrics
        self.countdown = self.patience
        self.verbose = verbose

        super(AccuracyTerminationCriteriaCallback, self).__init__()

    def on_epoch_end(self, batch, logs=None):

        if self.accuracy_metrics is None:
            self.accuracy_metrics = [metric for metric in logs.keys() if "accuracy" in metric]

        # We make sure that the all accuracies (e.g. the accuracy at all model outputs) are saturated.
        min_accuracy = min([logs[metric] for metric in self.accuracy_metrics])

        if min_accuracy == 1.0:
            self.countdown = self.countdown - 1
        else:
            self.countdown = self.patience

        if self.countdown == 0:
            self.model.stop_training = True
            if self.verbose:
                logging.info("\nAccuracy saturation callback triggered. Terminating training.\n")


class LossThresholdStoppingCallback(callbacks.Callback):
    """
    Callback for terminating training when loss reaches a pre-specified value.
    """

    def __init__(self, loss_threshold):

        self.loss_threshold = loss_threshold

        super(LossThresholdStoppingCallback, self).__init__()

    def on_batch_end(self, batch, logs=None):

        if logs.get('loss') < self.loss_threshold:
            logging.info("Loss of {} less than threshold of {}, terminating optimization.".format(logs.get('loss'),
                                                                                                   self.loss_threshold))
            self.model.stop_training =True


class DivergenceDetectorCallback(callbacks.Callback):
    """
    A callback for terminating training when optimization diverges immediately (usually due to excessive LR)
     using a simple heuristic.
    """

    def __init__(self, loss_ratio_threshold=1.05, patience=16):

        self.loss_ratio_threshold = loss_ratio_threshold
        self.patience = patience

        self.first_batch_loss = None
        self.current_loss = None
        self.num_triggers = 0

        super(DivergenceDetectorCallback, self).__init__()

    def on_train_begin(self, logs=None):

        pass

    def on_batch_end(self, batch, logs=None):

        if self.first_batch_loss is None:
            self.first_batch_loss = logs.get('loss')

        self.current_loss = logs.get('loss')

        if self.current_loss > self.first_batch_loss * self.loss_ratio_threshold:
            self.num_triggers += 1
            if self.num_triggers == self.patience:
                logging.info("\n==Divergence detected - first batch loss: %s, current loss: %s==\n",
                             self.first_batch_loss, self.current_loss)
                self.model.stop_training = True
        else:
            self.num_triggers = 0


class TrainingTimerCallback(callbacks.Callback):
    """
    Callback for timing various steps which keras does not report e.g. the delay at the beginning of training.
    """
    def __init__(self):

        self.train_start_timer = None
        self.first_batch_time = None

        super(TrainingTimerCallback, self).__init__()

    def on_train_begin(self, logs=None):
        self.train_start_timer = timeit.default_timer()

    def on_train_batch_end(self, batch, logs=None):
        # With linger timescale models we see a long delay at the first batch, so we time this specifically
        if self.first_batch_time is None:
            self.first_batch_time = timeit.default_timer() - self.train_start_timer
            logging.info("Time from train start to end of first batch was {} seconds".format(self.first_batch_time))
