import numpy as np

def logsig(xi, lam=1):
    # OMo: Nestačí toto?:
    #
    # xi = np.array(xi)
    # return 1 / (1 + np.exp(-lam*xi))
    
    if xi.ndim <= 1:
        return 1 / (1 + np.exp(-lam*xi))
    else:
        first = True
        for vec in xi:
            if first:
                output = 1 / (1 + np.exp(-lam*vec))
                first = False
            else:
                output = np.vstack((output, 1 / (1 + np.exp(-lam*vec))))
    return output


def net_out(w, x, lam=1):
    for layer in w:
        x = x.tolist()
        x_extended = np.hstack((x, np.ones((1))))
        x = logsig(np.dot(x_extended, layer), lam)
    return x


def net_size(arch):
    total = 0
    for i in range(len(arch)-1):
        total += (arch[i] + 1) * arch[i+1]
    return total


def vec_to_net(vec, arch, coef):
    # vec is a 1D vector of weights
    # return network as a list of arrays containing weights corresponding to the architecture arch
    vec = vec.flatten()
    net = []
    num = 0
    for i in range(len(arch) - 1):
        new_num = num + (arch[i] + 1) * arch[i+1]
        layer = coef * np.array(vec[num:new_num]).reshape(arch[i]+1, arch[i+1])
        net.append(layer)
        num = new_num
    return net

def net_to_file(net, arch, filepath):
    with open(filepath, "w") as f:
        f.write("\t".join(map(str, arch)) + "\n")
        for layer in net:
            height, width = layer.shape
            for y in range(height):
                f.write("\t".join([str(layer[y, x]) for x in range(width)]) + "\n")


def file_to_net(filepath):
    with open(filepath, "r") as f:
        arch = list(map(int, f.readline().split("\t")))
        net = []
        prev = None
        for layerSize in arch:
            if prev is not None:
                height = prev + 1
                width = layerSize
                layer = np.empty([height, width])
                for y in range(height):
                    layer[y, :] = list(map(float, f.readline().split("\t")))
                net.append(layer)
            prev = layerSize
    return net, arch



