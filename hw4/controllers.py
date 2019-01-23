import numpy as np
from cost_functions import trajectory_cost_fn
import time

class Controller():
	def __init__(self):
		pass

	# Get the appropriate action(s) for this state(s)
	def get_action(self, state):
		pass


class RandomController(Controller):
	def __init__(self,env):
		self.env = env
		super().__init__()

	def get_action(self, state):
		""" randomly sample an action uniformly from the action space """
		return self.env.action_space.sample()

class MPCcontroller(Controller):
	""" Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
	def __init__(self,env,dyn_model,horizon=5,cost_fn=None,num_simulated_paths=10):
		super().__init__()
		self.env = env
		self.dyn_model = dyn_model
		self.horizon = horizon
		self.cost_fn = cost_fn
		self.num_simulated_paths = num_simulated_paths

	def get_action(self, state):
		""" batch simulations through the model for speed """
		rand_cont = RandomController(self.env)

		obs, actions, next_obs = [],[],[]
		new_obs = np.repeat(state[np.newaxis,:],self.num_simulated_paths,0)

		for _ in range(self.horizon):
			new_actions = [rand_cont.get_action(state) for _ in range(self.num_simulated_paths)]
			obs.append(new_obs)
			new_obs = self.dyn_model.predict(obs,actions)
			next_obs.append(new_obs)
			actions.append(new_actions)

		obs,actions,next_obs = np.array(obs),np.array(actions),np.array(next_obs)
		costs = trajectory_cost_fn(self.cost_fn,obs,actions,next_obs)

		best_path_id = np.argmin(costs)
		return actions[0][best_path_id]