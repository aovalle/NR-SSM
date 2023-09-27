import os
import pickle
import joblib
import torch

MODEL_FILE = "model.pth"
BUFFER_FILE = "buffer.pth"
TEST_RESULTS_FILE = "test_results.csv"


def init_dirs(logdir):
    videodir = os.path.join(logdir, "videos")
    monitordir = os.path.join(logdir, "monitoring")
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(videodir, exist_ok=True)
    os.makedirs(monitordir, exist_ok=True)


def load_model(logdir, filename=MODEL_FILE):
    model_path = os.path.join(logdir, filename)
    return torch.load(model_path)

def save_model(dir, dict, filename=MODEL_FILE):
    if not os.path.exists(dir):
        os.makedirs(dir)
    path = os.path.join(dir, filename)

    torch.save(dict, path)
    print("Saved _models_ at `{}`".format(path))


def load_buffer(logdir, jlib=True):
    buffer_path = os.path.join(logdir, BUFFER_FILE)
    if jlib:
        return joblib.load(buffer_path)
    else:
        return _load_pickle(buffer_path)

def save_buffer(logdir, buffer, jlib=True):
    path = os.path.join(logdir, BUFFER_FILE)
    if jlib:
        joblib.dump(buffer, path, compress='zlib')
    else:
        _save_pickle(path, buffer)
    print("Saved _buffer_ at path `{}`".format(path))

def _load_pickle(path):
    with open(path, "rb") as pickle_file:
        data = pickle.load(pickle_file)
    return data


def _save_pickle(path, obj):
    with open(path, "wb") as pickle_file:
        pickle.dump(obj, pickle_file, protocol=4)


def save_test_results(logdir, msg):
    path = os.path.join(logdir, TEST_RESULTS_FILE)
    with open(path, "a+") as f:
        f.write(msg)