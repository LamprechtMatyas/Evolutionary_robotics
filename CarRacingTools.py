import numpy as np

import myNN


def find_beginning_of_car(arr):
    """
    Finds first bottom-left (?) pixel concaining car.
    """
    is_car = (arr[:,:,0] == 204) & (arr[:,:,1] == 0) & (arr[:,:,2] == 0)

    value_i = np.amax(is_car, axis=1)      # [i,j] -> for [i] we get True if there is j with car
    first_i = np.argmax(value_i)           # we get first i with True value (or 0 if all is False)
    first_j = np.argmax(is_car[first_i,:]) # we get corresponding j

    if not is_car[first_i, first_j]:       # check if it actually is car
         first_i, first_j = -1, -1

    return first_i, first_j


def state_to_track(arr):
    """
    Detects pixels containing track.
    """
    return np.amin((arr > 98) & (arr < 120), axis=2).astype(np.int8)


class InputTransformation:
    """
    Base class for transformation: observation -> NN input.
    """
    def __init__(self, input_vector_size):
        self.input_vector_size = input_vector_size

    def __call__(self, observation):
        pass

class CarBoxTransformation(InputTransformation):
    """ 
    Input transformation: car bounding box + track detection.
    """
    def __init__(self, y_min=-6, y_max=-1, x_min=-5, x_max=+7):
        assert y_min < y_max
        assert x_min < x_max

        super().__init__((x_max - x_min) * (y_max - y_min))
        self.y_min = y_min
        self.y_max = y_max
        self.x_min = x_min
        self.x_max = x_max

    def __call__(self, observation):
        y, x = find_beginning_of_car(observation)
        if y != -1:
            box = observation[(y+self.y_min):(y+self.y_max), (x+self.x_min):(x+self.x_max), :]
            return state_to_track(box).reshape(-1)
        return None

class DownsamplingTransformation(InputTransformation):
    """
    Input transformation: downsampling + track detection.
    """
    def __init__(self, stride):
        assert stride > 0 and stride <= 96
        super().__init__((96 // stride) ** 2)
        self.stride = stride
        self.slices = slice(self.stride // 2, 96, self.stride)

    def __call__(self, observation):
        return state_to_track(observation[self.slices, self.slices, :]).reshape(-1)

class OutputTransformation:
    """
    Base class for transformation: NN output -> action.
    """
    def __init__(self, output_vector_size, default_action):
        self.output_vector_size = output_vector_size
        self.default_action = default_action

    def __call__(self, output):
        pass

class ContinuousActionTransformation(OutputTransformation):
    """
    Output transformation: NN output = action (with left/right normalization).
    """
    def __init__(self, default_action=None):
        if default_action is None:
            default_action = [0, 0.5, 0]
        super().__init__(3, default_action)

    def __call__(self, output):
        if output is None:
            return self.default_action

        output[0] = output[0] * 2 - 1
        #if (output[0] > -0.05) and (output[0] < 0.05):
        #    output[0] = 0
        #if ((last_action < 0) and (output[0] > 0)) or ((last_action > 0) and (output[0] < 0)):
        #    total_reward -= 0.02
        return output

class DiscreteActionTransformation(OutputTransformation):
    """
    Output transformation: NN output = action logits -> argmax.
    """
    default_actions = [
        [ 0, 0, 0.0],
        [-1, 0, 0.0],
        [+1, 0, 0.0],
        [ 0, 1, 0.0],
        [ 0, 0, 0.8],
    ]

    def __init__(self, actions=None):
        if actions is None:
            actions = DiscreteActionTransformation.default_actions
        super().__init__(len(actions), DiscreteActionTransformation.default_actions[0])
        self.actions = actions

    def __call__(self, output):
        if output is None:
            return self.default_action

        return self.actions[np.argmax(output)]


class CarRacingAgentArchitecture:
    """
    Architecture of car racing agent.
    """
    def __init__(self, input_transformation, output_transformation, hidden_layer_size, weight_coef, logsig_lambda):
        self.input_transformation = input_transformation
        self.output_transformation = output_transformation
        self.hidden_layer_size = hidden_layer_size
        self.weight_coef = weight_coef
        self.logsig_lambda = logsig_lambda
        self.nn_arch = [self.input_transformation.input_vector_size, self.hidden_layer_size, self.output_transformation.output_vector_size]

    def action_fn(self, net, observation):
        net_input = self.input_transformation(observation)
        net_output = None if net_input is None else myNN.net_out(net, net_input, self.logsig_lambda)
        return self.output_transformation(net_output)

    def vec_to_net(self, vec):
        return myNN.vec_to_net(vec, self.nn_arch, self.weight_coef)