import torch
from rl_games.algos_torch.models import BaseModel, BaseModelNetwork
import numpy as np
from rl_games.algos_torch.running_mean_std import RunningMeanStd
import torch.nn as nn

normalize_keys = ["policy"] # Don't normalize the image

class MyRunningMeanStdObs(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        assert(isinstance(insize, dict))
        super(MyRunningMeanStdObs, self).__init__()
        self.running_mean_std = nn.ModuleDict({
            k : RunningMeanStd(v, epsilon, per_channel, norm_only) for k,v in insize.items() if k in normalize_keys
        })
    
    def forward(self, input, denorm=False):
        res = {}
        for k,v in input.items():
            if k in normalize_keys:
                res[k] = self.running_mean_std[k](v, denorm)
            else:
                res[k] = v
        return res
    
class BaseModelNetwork(nn.Module):
    def __init__(self, obs_shape, normalize_value, normalize_input, value_size):
        nn.Module.__init__(self)
        self.obs_shape = obs_shape
        self.normalize_value = normalize_value
        self.normalize_input = normalize_input
        self.value_size = value_size

        if normalize_value:
            self.value_mean_std = RunningMeanStd((self.value_size,)) #   GeneralizedMovingStats((self.value_size,)) #   
        if normalize_input:
            assert isinstance(obs_shape, dict)
            self.running_mean_std = MyRunningMeanStdObs(obs_shape)
           

    def norm_obs(self, observation):
        with torch.no_grad():
            return self.running_mean_std(observation) if self.normalize_input else observation

    def denorm_value(self, value):
        with torch.no_grad():
            return self.value_mean_std(value, denorm=True) if self.normalize_value else value
        
class MyA2CContinuousLogStd(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network

        def is_rnn(self):
            return self.a2c_network.is_rnn()
        
        def is_flow(self):
            return self.a2c_network.is_flow()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)

                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'values' : value,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }  
                if(self.is_flow()):
                    result['flow_entropy'] = self.calc_flow_entropy(input_dict.get('flow_samples'))
              
                return result
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.denorm_value(value),
                    'actions' : selected_action,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                return result
        def calc_flow_entropy(self, flow_samples):
            return -self.a2c_network.flow().log_prob(flow_samples).mean()
        
        def neglogp(self, x, mean, std, logstd):
            return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
                + logstd.sum(dim=-1)
