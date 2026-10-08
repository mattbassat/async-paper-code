"""Loading of stored simulation results."""
import pickle


def load_pickle(path):
    """Load one pickled result file."""
    with open(path, "rb") as f:
        return pickle.load(f)
