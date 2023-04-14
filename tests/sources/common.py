import pickle


def ensure_pickle(source):
    # Pickling support
    assert source == pickle.loads(pickle.dumps(source))
