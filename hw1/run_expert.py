#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20
Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""
# "run_expert.py" script code by Jonathan Ho (hoj@openai.com) modified and partially moved to
# RlAgent.gather_experience method in "rl_agent.py"

import argparse
from agents.rl_agent import ExpertAgent


def run_expert(envs,exps_rollouts,render=False,max_timesteps=None,cout=False,record=False):
    assert len(envs) == len(exps_rollouts),"argument lists must be the same length"
    agent = ExpertAgent(output_dir='experts_experience')
    for env_name, num_rollouts in zip(envs,exps_rollouts):
        agent.generate_experience(env_name,'experts/'+env_name+'.pkl',num_rollouts,render,max_timesteps,cout=cout,record=record)

def run_expert_all():
    envs_list = ['Ant-v1','HalfCheetah-v1','Hopper-v1','Humanoid-v1','Reacher-v1','Walker2d-v1']
    expert_rollouts_list = [10,10,10,10,10,10]
    run_expert(envs_list,expert_rollouts_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--envnames' ,nargs="*" ,type=str , default=['Ant-v1', ])
    parser.add_argument('-nr','--nums_rollouts' , nargs="*" , type=int ,default=[10, ])
    parser.add_argument('-r','--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('-co','--cout_rollouts',action='store_true',help='display rollouts progress printed instead of progressbar')
    parser.add_argument('-rec','--record',action='store_true',help='record rollouts videos')
    args = parser.parse_args()

    run_expert(args.envnames,args.nums_rollouts,args.render,args.max_timesteps,args.cout_rollouts,args.record)
