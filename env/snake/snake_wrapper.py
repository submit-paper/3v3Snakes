
from .chooseenv import make
from copy import deepcopy
from .common import *



class SnakeEnvWrapper():
    def __init__(self, mode='OneVsOne-v0'):
        self.env = make("snakes_3v3", conf=None)
        self.states = None
        self.ctrl_agent_index = [0, 1, 2]
        self.obs_dim = 26
        self.height = self.env.board_height
        self.width = self.env.board_width
        self.episode_reward = np.zeros(6)

    def act(self):
        return self.env.act(self.states)

    def reset(self):
        states = self.env.reset()
        length = []
        obs = process_obs_joint(states[0])
        self.states = deepcopy(states)
                        
        legal_action = get_legal_actions(states[0])
        info = {}
        info ["legal_action"] = legal_action

        return obs, info

    def step(self, actions):
        next_state, reward, done, _, info = self.env.step(self.env.encode(actions))
        next_obs = process_obs_joint(next_state[0])
        # reward shaping
        step_reward = get_reward_joint(reward, next_state[0], done)
        length = []
        for i in range(2,8):
            length.append(len(next_state[0][i]))
        info ["length"] = length
        legal_action = get_legal_actions(next_state[0])
        info ["legal_action"] = legal_action
        return next_obs, step_reward, done, info

    def render(self):
        self.env.render()


