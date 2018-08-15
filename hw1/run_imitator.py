import argparse
import os
import agents.rl_agent
from util import mkdir_rel

def run_imitator(cloner,envs,nums_rollouts,imits_epochs,render=False,pbar=False,cout=False,record=False):
    assert len(envs) == len(nums_rollouts),"argument lists must be the same length"
    assert cloner==True, "must pass at least one imitator agent flag from the following: -c"
    inputs = zip(envs,['experts_experience/'+env+'-'+str(nums_rollouts[i])+'_rollouts.pkl' for i,env in enumerate(envs)],nums_rollouts,imits_epochs)
    results = []
    for env_name, expert_data, num_rollouts, num_epochs in inputs:
        if cloner:
            cloner_output_dir = os.path.join('imitators_data',env_name,'cloner',''+str(num_rollouts)+'_exp_rollouts')
            mkdir_rel(cloner_output_dir)
            cloner = agents.rl_agent.CloningAgent(cloner_output_dir)
            epoch, bc_mean, bc_std, ex_mean, ex_std = cloner.clone_expert(env_name,expert_data,restore=True,render=render,
                                                                          num_epochs=num_epochs,num_test_rollouts=10,
                                                                          progbar=pbar,cout=cout,record=record)
            results.append((env_name, ex_mean, ex_std, bc_mean, bc_std))
            with open('results.txt'.format(epoch),'a') as f:
                for env_name , ex_mean , ex_std , bc_mean , bc_std in results:
                    f.writelines("Env = {0}, Expert: mean(std)={1:.5f}({2:.5f}), Cloner: mean(std)={3:.5f}({4:.5f}) exp_rolls={5} n_epochs={6}\n"
                                 .format(env_name, ex_mean , ex_std , bc_mean , bc_std, num_rollouts, epoch))

def run_imitator_all():
    envs_list = ['Ant-v1','HalfCheetah-v1','Hopper-v1','Humanoid-v1','Reacher-v1','Walker2d-v1']
    expert_rollouts_list = [10,10,10,10,10,10]
    epochs_list = [50,50,50,50,50,50]
    run_imitator(True,envs_list , expert_rollouts_list,epochs_list,False,False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--cloner',action='store_true',help='run cloner')
    parser.add_argument('-e','--envnames' ,nargs="*" ,type=str , default=['Ant-v1', ])
    parser.add_argument('-nr','--nums_rollouts' , nargs="*" , type=int ,default=[10, ])
    parser.add_argument('-ne','--nums_epochs' , nargs="*" , type=int ,default=[50, ])
    parser.add_argument('-r','--render', action='store_true')
    parser.add_argument('-pb','--progressbar',action='store_true',help='display epochs progressbar')
    parser.add_argument('-co','--cout_rollouts',action='store_true',help='display rollouts progress printed instead of progressbar')
    parser.add_argument('-rec','--record',action='store_true',help='record rollouts videos')
    args = parser.parse_args()

    run_imitator(args.cloner,args.envnames,args.nums_rollouts,args.nums_epochs,args.render,args.progressbar,args.cout_rollouts,args.record)
    # main()