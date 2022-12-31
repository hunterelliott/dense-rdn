import os
import logging
import sys
import inspect
import subprocess
import timeit

from PIL import Image
from types import ModuleType, FunctionType, MethodType

import numpy as np

from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model as keras_load_model


def get_n_t_out_directory(out_dir_base, n_t):

    return os.path.join(out_dir_base, 'Nt' + str(n_t))


def get_model_out_file(out_dir):
    return os.path.join(out_dir, 'model.h5')


def save_model(model, out_dir):
    out_path = get_model_out_file(out_dir)
    logging.info("Saving model to {}".format(out_path))
    model.save(out_path, save_format='h5')


def load_model(out_dir):
    """
    A wrapper to handle custom objects etc. when loading keras models.
    Args:
        out_dir: the directory the model was output to

    Returns:
        model: A keras Model object.

    """

    model_file = get_model_out_file(out_dir)
    logging.info("Loading model from {}".format(model_file))
    model = keras_load_model(model_file, compile=False)

    return model


def prep_output_dir(dir, force=False):
    if not os.path.exists(dir):
        os.makedirs(dir)
    elif os.path.exists(get_log_file_from_dir(dir)) and not force:
        raise AssertionError("The specified output directory already contains a log file! Use force=True to append.")


def get_commit_hash():

    return subprocess.run(['git rev-parse HEAD'], shell=True, capture_output=True).stdout.strip().decode('UTF-8')


def log_uncommitted_changes(log_dir):

    changes = subprocess.run(['git diff HEAD'], shell=True, capture_output=True).stdout.strip().decode('UTF-8')
    changes_path = os.path.join(log_dir, 'uncommitted_changes.txt')
    logging.info("Logging any uncommited changes to {}".format(changes_path))
    with open(changes_path, 'w') as changes_file:
        print(changes, file=changes_file)


def posterity_log(log_dir, local_vars, script, force=False):
    """
    Wrapper to configure logging and then log various info useful for reproducing experiments.
    I know, it's all hacky but this is not a software engineering project...
    """

    prep_output_dir(log_dir, force=force)
    configure_logging(log_dir)
    logging.info("----Initializing, Logging State----")
    logging.info("Running {}".format(script))
    logging.info("From commit hash {}".format(get_commit_hash()))
    log_uncommitted_changes(log_dir)
    log_local_vars(local_vars)


def configure_logging(log_dir):

    # Clear existing loggers before configuring as they can prevent proper logging to file
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=get_log_file_from_dir(log_dir), format='%(asctime)s : %(message)s',
                        level=logging.INFO, filemode='a')
    # Make sure our logs go to stdout as well
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    root = logging.getLogger()
    root.addHandler(stdout_handler)


def get_log_file_from_dir(log_dir):
    return os.path.join(log_dir, 'log.txt')


def log_local_vars(local_vars):
    """
    A hack to allow me to define parameters in a script and still have them in the log as well
    """
    local_vars = {key: value for key, value in local_vars.items() if
                not isinstance(value, (ModuleType, FunctionType, MethodType)) and not key[0] == '_' and not key in (
                'In', 'Out', 'exit', 'quit', 'Model', 'root', 'stdout_handler', 'FunctionType', 'MethodType',
                'ModuleType', 'all_vars') and not 'Layer' in key and not is_tensorflow_object(value)}
    logging.info("Parameters:")
    logging.info(local_vars)


def log_and_time(message_or_time):
    """
    Tool for logging the time of blocks of code. First call should be just a message and will return t_start.
    Second call should input t_start and will log and return the elapsed time.

    Args:
        message_or_time: Message on first call, t_start on second call.

    Returns:
        t_start or t_elapsed

    """

    if isinstance(message_or_time, str):
        t = timeit.default_timer()
        logging.info(message_or_time)
    else:
        t = timeit.default_timer() - message_or_time
        logging.info("Finished. Took {} seconds.".format(str(t)))

    return t


def is_tensorflow_object(entity):
    # I know, I know...
    is_tf = False
    try:
        is_tf = 'tensorflow' in inspect.getfile(entity)
    except:
        is_tf = False

    return is_tf


def load_and_normalize_image(x_shape, image_path, clip_below=-1.0):
    """
    Loads an image to be used in an image similarity loss
    Args:
        x_shape: (height, width, n_chan) expected shape of model output
        image_path: path to image file
        clip_below: Optionally clip values below this e.g. to prevent negative concentrations.

    Returns:
        image: Normalized image with same (height, width) as x_shapoe
    """

    ref_image = Image.open(image_path)
    # Resize and normalize to zero mean unit variance
    ref_image = np.array(ref_image.resize(x_shape[0:2]), dtype=np.float32)
    ref_image = ref_image - np.mean(ref_image, axis=(0, 1), keepdims=True)
    ref_image = ref_image / np.std(ref_image, axis=(0, 1), keepdims=True)

    if clip_below is not None:
        ref_image[ref_image < clip_below] = clip_below

    return ref_image.astype(np.float32)


# ---- Old, not-refactored / retested below this line ---- #


class DebuggerCallback(callbacks.Callback):

    def __init__(self, loss_delta_threshold=1.00, monitor_weight=None):

        self.previous_loss = None
        self.current_loss = None
        self.monitor_weight = monitor_weight
        self.loss_delta_threshold = loss_delta_threshold

        super(DebuggerCallback, self).__init__()

    def on_train_begin(self, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):

        self.previous_loss = self.current_loss
        self.current_loss = logs.get('loss')


        weight_val = K.eval(self.monitor_weight)
        adam_mean_val = K.eval(self.model.optimizer.weights[1])
        adam_var_val = K.eval(self.model.optimizer.weights[31])

        if np.sum(np.isnan(weight_val))  > 0:
            print("========= NAN in weight!!==============")
        if np.sum(np.isnan(adam_mean_val))  > 0:
            print("========= NAN in weight!!==============")
        if np.sum(np.isnan(adam_var_val))  > 0:
            print("========= NAN in weight!!==============")

        print("Weight max: {}, min: {}, adam mean max: {}, min: {}, adam var max: {}, min: {}".format(
            weight_val.max(), weight_val.min(), adam_mean_val.max(), adam_mean_val.min(), adam_var_val.max(), adam_var_val.min()))

        if self.previous_loss:
            loss_delta = self.current_loss - self.previous_loss
            if loss_delta > self.loss_delta_threshold:
                print(" Loss increased by {}".format(loss_delta))

    def set_model(self, model):
        self.model = model
        if self.monitor_weight:
            self.grads = model.optimizer.get_gradients(model.total_loss, self.monitor_weight)



