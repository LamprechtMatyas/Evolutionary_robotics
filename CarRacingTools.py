import numpy as np
# import time

def find_beginning_of_car(arr):
    # start = time.time() # OMo: Původně jsem si nemyslel, že toto bude rychlejší, ale je
    
    is_car = (arr[:,:,0] == 204) & (arr[:,:,1] == 0) & (arr[:,:,2] == 0)

    value_i = np.amax(is_car, axis=1)      # [i,j] -> for [i] we get True if there is j with car
    first_i = np.argmax(value_i)           # we get first i with True value (or 0 if all is False)
    first_j = np.argmax(is_car[first_i,:]) # we get corresponding j

    if not is_car[first_i, first_j]: # check if it actually is car
         first_i, first_j = -1, -1

    """
    print("A", time.time() - start) # 0.0 - 0.001

    start = time.time()

    for i in range(len(arr)//2, len(arr)):
        for j in range(len(arr[i])):
            if (arr[i][j][0] == 204) & (arr[i][j][1] == 0) & (arr[i][j][2] == 0):
                assert i == first_i
                assert j == first_j
                print("B", time.time() - start)
                return i, j
    assert first_i == -1
    assert first_j == -1
    print("B", time.time() - start) # 0.15 - 0.30
    return -1, -1
    """

    return first_i, first_j


def state_to_track(arr):
    track_arr = np.amin((arr > 98) & (arr < 120), axis=2).astype(np.int8)

    """
    new_arr = np.zeros(shape=(len(arr), len(arr[0])))

    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if (arr[i][j][0] > 98) & (arr[i][j][0] < 120) & (arr[i][j][1] > 98) & (arr[i][j][1] < 120) & (arr[i][j][2] > 98) & (arr[i][j][2] < 120):
                new_arr[i][j] = 1
            else:
                new_arr[i][j] = 0

    assert np.sum(is_grey - new_arr) == 0
    """

    return track_arr