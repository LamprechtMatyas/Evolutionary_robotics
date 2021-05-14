import numpy as np

ARCH = [9, 4]

def logsig(xi):
    if xi.ndim <= 1:
        return 1 / (1 + np.exp(-0.3*xi))
    else:
        first = True
        for vec in xi:
            if first is True:
                output = 1 / (1 + np.exp(-0.3*vec))
                first = False
            else:
                output = np.vstack((output, 1 / (1 + np.exp(-0.3*vec))))
    return output


def net_out(w, x):
    for layer in w:
        x = x.tolist()
        x_extended = np.hstack((x, np.ones((1))))
        x = logsig(np.dot(x_extended, layer))
    return x


def net_size(arch=ARCH):
    total = 0
    for i in range(len(arch)-1):
        total += (arch[i] + 1) * arch[i+1]
    return total


def vec_to_net(vec, arch=None, coef=None):
    # vec is a 1D vector of weights
    # return network as a list of arrays containing weights corresponding to the architecture arch
    vec = vec.flatten()
    global ARCH
    if arch == None:
        arch = ARCH
    global COEF
    if coef == None:
        coef = COEF
    net = []
    num = 0
    for i in range(len(arch) - 1):
        new_num = num + (arch[i] + 1) * arch[i+1]
        layer = coef*np.array(vec[num:new_num]).reshape(arch[i]+1, arch[i+1])
        net.append(layer)
        num = new_num
    return net




