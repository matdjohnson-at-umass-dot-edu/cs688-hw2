
import numpy as np


class ParamHolder:

    def __init__(self):
        self.feature_params = np.array([])
        self.feature_grad = np.array([])
        self.transition_params = np.array([])
        self.transition_grad = np.array([])

    def load_params(self):
        feature_params_file_object = open('./model/feature-params.txt', 'r')
        feature_params_list = list()
        for line in feature_params_file_object:
            feature_params_list.append(np.array(line.strip().split(' '), dtype=np.float128))
        self.feature_params = np.stack(feature_params_list)
        assert (np.shape(self.feature_params) == (10, 321))
        feature_grad_file_object = open('./model/feature-grad.txt', 'r')
        feature_grad_list = list()
        for line in feature_grad_file_object:
            feature_grad_list.append(np.array(line.strip().split(' '), dtype=np.float128))
        self.feature_grad = np.stack(feature_grad_list)
        assert (np.shape(self.feature_grad) == (10, 321))
        transition_params_file_object = open('./model/transition-params.txt', 'r')
        transition_params_list = list()
        for line in transition_params_file_object:
            transition_params_list.append(np.array(line.strip().split(' '), dtype=np.float128))
        self.transition_params = np.stack(transition_params_list)
        assert (np.shape(self.transition_params) == (10, 10))
        transition_grad_file_object = open('./model/transition-grad.txt', 'r')
        transition_grad_list = list()
        for line in transition_grad_file_object:
            transition_grad_list.append(np.array(line.strip().split(' '), dtype=np.float128))
        self.transition_grad = np.stack(transition_grad_list)
        assert (np.shape(self.transition_grad) == (10, 10))

    def get_feature_params(self):
        return_feature_params = np.copy(self.feature_params)
        return return_feature_params

    def get_feature_grad(self):
        return_feature_grad = np.copy(self.feature_grad)
        return return_feature_grad

    def get_transition_params(self):
        return_transition_params = np.copy(self.transition_params)
        return return_transition_params

    def get_transition_grad(self):
        return_transition_grad = np.copy(self.transition_grad)
        return return_transition_grad

