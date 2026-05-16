
import gym
import numpy as np

class ExampleEnv(gym.Env):
    '''
    Just example empty env which demonstrates additional features compared to the default openai gym
    '''
    def __init__(self,  **kwargs):
        gym.Env.__init__(self)

        self.use_central_value = True
        self.value_size = 1
        self.concat_infos = False
        self._curr_steps=0
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(2, ), dtype=np.float32) # gym.spaces.Discrete(3), gym.spaces.Box(low=0, high=1, shape=(3, ), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(6, ), dtype=np.float32) # or Dict

    def get_number_of_agents(self):
        return 1

    def has_action_mask(self):
        return False


    def reset(self):
        self._curr_steps=0
        obses = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype(np.float32)
        return {"obs": obses, "state": obses}

    def step(self, action):
        info = {}  
        self._curr_steps += 1
        reward = np.array(action[0]).astype(np.float32)
        obses = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype(np.float32)
        done = False
        if(self._curr_steps>10):
            self._curr_steps = 0
            done = True
        return {"obs": obses, "state": obses}, reward, done, info

    def get_action_mask(self):
        pass