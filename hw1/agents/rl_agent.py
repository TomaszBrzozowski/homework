
class RlAgent:
    def __init__(self,output_dir=None):
        self.experience = None
        self.policy_fn = None
        self.output_dir = output_dir
        self.model = self.get_model()
        self.timestep = None
        self.episode = None
    def gather_experience(self,env_name,expert_policy_filename=None,render=None,max_timesteps=None,num_rollouts=None,output_dir=None):
        raise NotImplementedError
    def save_experience(self,output_dir,filename):
        raise NotImplementedError
    def save_model(self):
        raise NotImplementedError
    def restore_model(self):
        raise NotImplementedError
    def get_model(self):
        raise NotImplementedError

class ExpertAgent(RlAgent):
    def gather_experience(self,env_name,expert_policy_filename=None,render=None,max_timesteps=None,num_rollouts=None,output_dir=None):
        pass
    def save_experience(self,output_dir,filename):
        pass
    def load_policy(self,expert_policy_filename):
        pass
    def save_model(self):
        pass
    def restore_model(self):
        pass
    def get_model(self):
        pass

class CloningAgent(RlAgent):
    def save_experience(self,output_dir,filename):
        pass
    def load_experience(self,expert_experience_filename):
        pass
    def gather_experience(self):
        pass
    def clone_expert_bahavior(self):
        pass

class BehaviorCloningAgent(CloningAgent):
    def gather_experience(self):
        pass
    def clone_expert_bahavior(self):
        pass

class DaggerAgent(CloningAgent):
    def gather_experience(self):
        pass
    def clone_expert_bahavior(self):
        pass