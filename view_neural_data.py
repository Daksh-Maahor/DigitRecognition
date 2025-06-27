import pickle
import numpy as np
import pathlib

def get_neural_data():
    with open(f"{pathlib.Path(__file__).parent.absolute()}/data/neural_data.dat", "rb") as f:
        w_i_h = pickle.load(f)
        b_i_h = pickle.load(f)
        w_h_o = pickle.load(f)
        b_h_o = pickle.load(f)
    
    return w_i_h, b_i_h, w_h_o, b_h_o

if __name__ == "__main__":
    a, b, c, d = get_neural_data()
    print(a)
    print(b)
    print(c)
    print(d)
