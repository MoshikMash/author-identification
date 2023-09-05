import pickle
import numpy as np


def generate_cm(similarity_matrices, numer_of_matrices):
    result = np.mean(similarity_matrices[:numer_of_matrices], axis=0)
    return result
