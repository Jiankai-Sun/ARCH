import torch
from rl_games.algos_torch.network_builder import NetworkBuilder
import torch.nn as nn
from rl_games.algos_torch import torch_ext
import zuko

class MyNetworkBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shapes = kwargs.pop('input_shape')
            print('input_shapes: ', input_shapes)
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)
            self.permute_input = num_seqs = kwargs.pop('permute_input', True)

            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params, input_shapes)
            # print('params: ', params)
            self.actor_cnn = nn.Sequential()
            self.critic_cnn = nn.Sequential()
            self.actor_mlp = nn.Sequential()
            self.critic_mlp = nn.Sequential()
            
            if(self.has_flow):
                self.flow = zuko.flows.NSF(features=2, transforms=3, hidden_features=(64, 64))
            else:
                self.flow = None
                
            cnn_output_size = 0
            input_shape = 0
            print('[my_network] input_shapes: ', input_shapes)
            if self.has_cnn:
                cnn_input_shape = input_shapes["image"]
                # print('cnn_input_shape: ', cnn_input_shape)  # (60, 60, 1)
                if self.permute_input:
                    cnn_input_shape = torch_ext.shape_whc_to_cwh(cnn_input_shape)
                cnn_args = {
                    'ctype' : self.cnn['type'], 
                    'input_shape' : cnn_input_shape, 
                    'convs' :self.cnn['convs'], 
                    'activation' : self.cnn['activation'], 
                    'norm_func_name' : self.normalization,
                }
                self.actor_cnn = self._build_conv(**cnn_args)

                if self.separate:
                    self.critic_cnn = self._build_conv( **cnn_args)

                cnn_output = nn.Sequential(*self.actor_cnn)(torch.rand(1, *(cnn_input_shape))).flatten(1)
                cnn_output_size = cnn_output.data.size(1)

                if self.align_modality_dim:
                    align_cnn_output_size = 16  # 64
                    mlp_args = {
                        'input_size': cnn_output_size,
                        'units': [cnn_output_size // 2, align_cnn_output_size],
                        'activation': self.activation,
                        'norm_func_name': self.normalization,
                        'dense_func': torch.nn.Linear,
                        'd2rl': self.is_d2rl,
                        'norm_only_first_layer': self.norm_only_first_layer
                    }
                    self.actor_image_align = self._build_mlp(**mlp_args)

            if "policy" in input_shapes:
                input_shape = input_shapes["policy"][0]
                if self.align_modality_dim:
                    mlp_args = {
                        'input_size': input_shape,
                        'units': [input_shape, 64],
                        'activation': self.activation,
                        'norm_func_name': self.normalization,
                        'dense_func': torch.nn.Linear,
                        'd2rl': self.is_d2rl,
                        'norm_only_first_layer': self.norm_only_first_layer
                    }
                    self.actor_policy_align = self._build_mlp(**mlp_args)
                    input_shape = 64
            if self.align_modality_dim:
                mlp_input_size = align_cnn_output_size + input_shape
            else:
                mlp_input_size = cnn_output_size+input_shape
            print("MLP Input shape: {}+{}={}".format(cnn_output_size, input_shape, mlp_input_size))

            if len(self.units) == 0:
                out_size = cnn_output_size
            else:
                out_size = self.units[-1]

            if self.has_rnn:
                if not self.is_rnn_before_mlp:
                    rnn_in_size = out_size
                    if self.rnn_concat_input:
                        rnn_in_size += mlp_input_size

                    out_size = self.rnn_units
                    if self.rnn_concat_output:
                        out_size += mlp_input_size
                else:
                    rnn_in_size = mlp_input_size

                    mlp_input_size = self.rnn_units
                    if self.rnn_concat_output:
                        mlp_input_size += mlp_input_size

                if self.separate:
                    self.a_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    self.c_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    if self.rnn_ln:
                        self.a_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                        self.c_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                else:
                    self.rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    if self.rnn_ln:
                        self.layer_norm = torch.nn.LayerNorm(self.rnn_units)

            mlp_args = {
                'input_size' : mlp_input_size,
                'units' : self.units, 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }
            self.actor_mlp = self._build_mlp(**mlp_args)
            if self.separate:
                self.critic_mlp = self._build_mlp(**mlp_args)

            self.value = self._build_value_layer(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            if self.is_discrete:
                self.logits = torch.nn.Linear(out_size, actions_num)
            '''
                for multidiscrete actions num is a tuple
            '''
            if self.is_multi_discrete:
                self.logits = torch.nn.ModuleList([torch.nn.Linear(out_size, num) for num in actions_num])
            if self.is_continuous:
                self.mu = torch.nn.Linear(out_size, actions_num)
                self.mu_act = self.activations_factory.create(self.space_config['mu_activation']) 
                mu_init = self.init_factory.create(**self.space_config['mu_init'])
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 
                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

                if self.fixed_sigma:
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                else:
                    self.sigma = torch.nn.Linear(out_size, actions_num)

            mlp_init = self.init_factory.create(**self.initializer)
            if self.has_cnn:
                cnn_init = self.init_factory.create(**self.cnn['initializer'])

            for m in self.modules():         
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)    

            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.fixed_sigma:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)  

            
        def forward(self, obs_dict):
            vec_obs = obs_dict.get("obs").get("policy", None)
            image_obs = obs_dict.get("obs").get("image", None)

            states = obs_dict.get('rnn_states', None)
            dones = obs_dict.get('dones', None)
            bptt_len = obs_dict.get('bptt_len', 0)

            if self.has_cnn:
                # for obs shape 4
                # input expected shape (B, W, H, C)
                # convert to (B, C, W, H)
                if self.permute_input and len(image_obs.shape) == 4:
                    image_obs = image_obs.permute((0, 3, 1, 2))

            assert not self.separate

            cnn_out = self.actor_cnn(image_obs)

            if self.align_modality_dim:
                vec_obs = self.actor_policy_align(vec_obs)
            # print('vec_obs.shape, cnn_out.shape: ', vec_obs.shape, cnn_out.shape)
            # vec_obs.shape, cnn_out.shape:  torch.Size([4, 9]) torch.Size([4, 256, 26, 26])
            if cnn_out is None:
                out = vec_obs
            else:
                cnn_out = cnn_out.flatten(1)
                if self.align_modality_dim:
                    cnn_out = self.actor_image_align(cnn_out)
                out = torch.cat([vec_obs, cnn_out], axis=1)
            # print('out.shape: ', out.shape)
            if self.has_rnn:
                seq_length = obs_dict.get('seq_length', 1)

                if not self.is_rnn_before_mlp:
                    out = self.actor_mlp(out)
                    if self.rnn_concat_input:
                        out = torch.cat([out, cnn_out], dim=1)

                batch_size = out.size()[0]
                num_seqs = batch_size // seq_length
                out = out.reshape(num_seqs, seq_length, -1)

                if len(states) == 1:
                    states = states[0]

                out = out.transpose(0, 1)
                if dones is not None:
                    dones = dones.reshape(num_seqs, seq_length, -1)
                    dones = dones.transpose(0, 1)
                out, states = self.rnn(out, states, dones, bptt_len)
                out = out.transpose(0, 1)
                out = out.contiguous().reshape(out.size()[0] * out.size()[1], -1)

                if self.rnn_ln:
                    out = self.layer_norm(out)
                if self.rnn_concat_output:
                    out = torch.cat([out, cnn_out], dim=1)
                if self.is_rnn_before_mlp:
                    out = self.actor_mlp(out)
                if type(states) is not tuple:
                    states = (states,)
            else:
                out = self.actor_mlp(out)
            value = self.value_act(self.value(out))

            if self.central_value:
                return value, states

            if self.is_discrete:
                logits = self.logits(out)
                return logits, value, states
            if self.is_multi_discrete:
                logits = [logit(out) for logit in self.logits]
                return logits, value, states
            if self.is_continuous:
                mu = self.mu_act(self.mu(out))
                if self.fixed_sigma:
                    sigma = self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(out))
                return mu, mu*0 + sigma, value, states
                    
        def is_separate_critic(self):
            return self.separate

        def is_rnn(self):
            return self.has_rnn
        
        def is_flow(self):
            return self.has_flow

        def get_default_rnn_state(self):
            if not self.has_rnn:
                return None
            num_layers = self.rnn_layers
            if self.rnn_name == 'identity':
                rnn_units = 1
            else:
                rnn_units = self.rnn_units
            if self.rnn_name == 'lstm':
                if self.separate:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)),
                            torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)))
                else:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)))
            else:
                if self.separate:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)))
                else:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)),)                

        def load(self, params, input_shapes):
            self.separate = params.get('separate', False)
            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.is_d2rl = params['mlp'].get('d2rl', False)
            self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_rnn = 'rnn' in params
            self.has_flow = 'flow' in params
            self.has_space = 'space' in params
            self.central_value = params.get('central_value', False)
            self.joint_obs_actions_config = params.get('joint_obs_actions', None)

            if self.has_space:
                self.is_multi_discrete = 'multi_discrete'in params['space']
                self.is_discrete = 'discrete' in params['space']
                self.is_continuous = 'continuous'in params['space']
                if self.is_continuous:
                    self.space_config = params['space']['continuous']
                    self.fixed_sigma = self.space_config['fixed_sigma']
                elif self.is_discrete:
                    self.space_config = params['space']['discrete']
                elif self.is_multi_discrete:
                    self.space_config = params['space']['multi_discrete']
            else:
                self.is_discrete = False
                self.is_continuous = False
                self.is_multi_discrete = False

            if self.has_rnn:
                self.rnn_units = params['rnn']['units']
                self.rnn_layers = params['rnn']['layers']
                self.rnn_name = params['rnn']['name']
                self.rnn_ln = params['rnn'].get('layer_norm', False)
                self.is_rnn_before_mlp = params['rnn'].get('before_mlp', False)
                self.rnn_concat_input = params['rnn'].get('concat_input', False)
                self.rnn_concat_output = params['rnn'].get('concat_output', False)

            if ('cnn' in params) and ("image" in input_shapes):
                self.has_cnn = True
                self.cnn = params['cnn']
                self.permute_input = self.cnn.get('permute_input', True)
            else:
                self.has_cnn = False

            self.align_modality_dim = 'align_modality_dim' in params and params['align_modality_dim'] and self.has_cnn

    def build(self, name, **kwargs):
        net = MyNetworkBuilder.Network(self.params, **kwargs)
        return net
