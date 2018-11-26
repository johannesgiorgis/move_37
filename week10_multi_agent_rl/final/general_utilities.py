import csv
import errno
import os
import pathlib
import json


def ensure_directory_exists(base_directory):
    """
    Makes a directory if it does not exist
    """
    try:
        os.makedirs(base_directory)
    except OSError as ex:
        if ex.errno != errno.EEXIST:
            raise ex


def load_dqn_weights_if_exist(dqns, weights_filename_prefix, weights_filename_extension=".h5"):
    """
    Loads weights if they exist, otherwise does nothing
    """
    for i, dqn in enumerate(dqns):
        # TODO should not work if only some weights available?
        dqn_filename = weights_filename_prefix + str(i) + weights_filename_extension
        if os.path.isfile(dqn_filename):
            print("Found old weights to use for agent {}".format(i))
            dqn.load(dqn_filename)


def save_dqn_weights(dqns, weights_filename_prefix, weights_filename_extension=".h5"):
    """
    Saves weights
    """
    p = pathlib.Path(weights_filename_prefix)
    if len(p.parts) > 1:
        dump_dirs = pathlib.Path(*p.parts[:-1])
        ensure_directory_exists(str(dump_dirs))
    for i, dqn in enumerate(dqns):
        dqn_filename = weights_filename_prefix + str(i) + weights_filename_extension
        dqn.save(dqn_filename)


def dump_dict_as_json(dict_to_dump, filename):
    p = pathlib.Path(filename)
    if len(p.parts) > 1:
        dump_dirs = pathlib.Path(*p.parts[:-1])
        ensure_directory_exists(str(dump_dirs))
    json_str = json.dumps(dict_to_dump)
    with open(filename, "w") as jsonfile:
        jsonfile.write(json_str)

