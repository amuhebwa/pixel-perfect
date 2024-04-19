"""
This script generates random sets of roads from the list of all roads in Kenya.
Run training/inference on different sets of roads ensures that the model is not biased towards a single set of
roads.
"""

import random
import code
import pprint
import numpy as np
all_roads = ['A1', 'A2', 'B1', 'B3', 'B5', 'B6', 'B9', 'C103', 'C16', 'C17',
             'C19', 'C20', 'C23', 'C25', 'C27', 'C28', 'C30', 'C37', 'C38',
             'C51', 'C58', 'C61', 'C63', 'C64', 'C65', 'C67', 'C68', 'C69',
             'C77', 'C78', 'C79', 'C82', 'C86', 'C88', 'C91', 'C92'
             ]

if __name__ == "__main__":
    no_of_roads = len(all_roads)
    n_times = 10
    no_of_elements_to_generate = [int(np.ceil(no_of_roads * 0.1)), int(np.ceil(no_of_roads * 0.2)), int(np.ceil(no_of_roads * 0.3)), int(np.ceil(no_of_roads * 0.4)),
    int(np.ceil(no_of_roads * 0.5)), int(np.ceil(no_of_roads * 0.6)), int(np.ceil(no_of_roads * 0.7)), int(np.ceil(no_of_roads * 0.8)), int(np.ceil(no_of_roads * 0.9))]
    res_dict = {}
    for _ind, n in enumerate(no_of_elements_to_generate):
        temp_arr = []
        for i in np.arange(n_times):
            results = random.sample(set(all_roads), n)
            temp_arr.append(results)
        res_dict.update({_ind+1: temp_arr})
        del temp_arr
    
    # code.interact(local=locals())
