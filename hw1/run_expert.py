#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20
Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""
# "run_expert.py" script code by Jonathan Ho (hoj@openai.com) moved to
# ExpertAgent.gather_experience method in "rl_agent.py"

import argparse
from agents.rl_agent import ExpertAgent


def run_expert():
    expert_inputs = [('Ant-v1', 'experts/Ant-v1.pkl', False, 10),
                     ('HalfCheetah-v1', 'experts/HalfCheetah-v1.pkl', False, 10),
                     ('Hopper-v1', 'experts/Hopper-v1.pkl', False, 10),
                     ('Humanoid-v1', 'experts/Humanoid-v1.pkl', False, 10),
                     ('Reacher-v1', 'experts/Reacher-v1.pkl', False, 10),
                     ('Walker2d-v1', 'experts/Walker2d-v1.pkl', False, 10)]

    agent = ExpertAgent(output_dir='experts_experience')
    for env_name, expert_policy_filename, render, num_rollouts in expert_inputs:
        agent.gather_experience(env_name, expert_policy_filename, render, num_rollouts=num_rollouts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20, help='Number of expert roll outs')
    args = parser.parse_args()

    agent = ExpertAgent(output_dir='experts_experience')
    agent.gather_experience(args.envname, args.expert_policy_file, args.render,
                            args.max_timesteps, args.num_rollouts)


if __name__ == '__main__':
    main()
