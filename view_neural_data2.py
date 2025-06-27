import pickle
import numpy as np
import pathlib

def get_neural_data():
    with open(f"{pathlib.Path(__file__).parent.absolute()}/data/neural_data_real.dat", "rb") as f:
        w_i_h1 = pickle.load(f)
        b_i_h1 = pickle.load(f)
        w_h1_h2 = pickle.load(f)
        b_h1_h2 = pickle.load(f)
        w_h2_o = pickle.load(f)
        b_h2_o = pickle.load(f)
    
    return w_i_h1, b_i_h1, w_h1_h2, b_h1_h2, w_h2_o, b_h2_o

if __name__ == "__main__":
    a, b, c, d, e, f = get_neural_data()
    print(a)
    print(b)
    print(c)
    print(d)
    print(e)
    print(f)
    input()
