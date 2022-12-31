"""
This module contains transition models - i.e. models of the form X_t+1 = f(X_t)
"""
import os
import logging

from analysis import get_compound_model_components
from utils import load_model

from .crn import *
from .generic import *
from .gray_scott import *
from .misc import *


def transition_model(model_name, freeze_model=False, **kwargs):
    """
    Wrapper for retrieving a specified transition model by name.
    Args:
        model_name: name of model to be retrieved from this module OR a directory of a composite model to load the
            transition model from.
        **kwargs: additional arguments to pass to the model constructor function

    Returns:
        model: a keras Model object.

    """
    if os.path.exists(model_name):
        logging.info("Loading transition model:")
        model = get_compound_model_components(load_model(model_name))[1]
    else:
        logging.info("Building transition model {}".format(model_name))
        model = globals()[model_name](**kwargs)

    if freeze_model:
        model.trainable = False

    return model


