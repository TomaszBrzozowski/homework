import argparse
import os
import agents.rl_agent
from util import mkdir_rel

def run_imitator(cloner,envs,exps_rollouts,imits_epochs,pbar=False,cout=False):
    assert len(envs) == len(exps_rollouts),"argument lists must be the same length"
    assert cloner==True, "must pass at least one imitator agent flag from the following: -c"
    inputs = [(env,'experts_experience/'+env+'-'+str(exp_rollouts)+'_rollouts.pkl',exp_rollouts,epoch)
              for env in envs for exp_rollouts in exps_rollouts for epoch in imits_epochs]
    results = []

    for env_name, expert_data, exp_rollouts, num_epochs in inputs:
        if cloner:
            cloner_output_dir = os.path.join('imitators_data',env_name,'cloner',''+str(exp_rollouts)+'_exp_rollouts')
            mkdir_rel(cloner_output_dir)
            cloner = agents.rl_agent.CloningAgent(cloner_output_dir)
            epoch, bc_mean, bc_std, ex_mean, ex_std = cloner.clone_expert(env_name,expert_data,restore=True,
                                                                          num_epochs=num_epochs,num_test_rollouts=10,progbar=pbar,cout=cout)
            results.append((env_name, ex_mean, ex_std, bc_mean, bc_std))
            with open('results.txt'.format(epoch),'a') as f:
                for env_name , ex_mean , ex_std , bc_mean , bc_std in results:
                    f.writelines("Env = {0}, Expert: mean(std)={1:.5f}({2:.5f}), Cloner: mean(std)={3:.5f}({4:.5f}) exp_rolls={5} n_epochs={6}\n"
                                 .format(env_name, ex_mean , ex_std , bc_mean , bc_std, exp_rollouts, epoch))

def run_imitator_all():
    envs_list = ['Ant-v1','HalfCheetah-v1','Hopper-v1','Humanoid-v1','Reacher-v1','Walker2d-v1']
    expert_rollouts_list = [10,10,10,10,10,10]
    epochs_list = [50,50,50,50,50,50]
    run_imitator(True,envs_list , expert_rollouts_list,epochs_list,False,False)

if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument('-c','--cloner',action='store_true',help='run cloner')
    CLI.add_argument('-e','--envs' ,nargs="*" ,type=str , default=['Ant-v1', ])
    CLI.add_argument('-er','--exp_rollouts' , nargs="*" , type=int ,default=[10, ])
    CLI.add_argument('-ne','--nums_epochs' , nargs="*" , type=int ,default=[50, ])
    CLI.add_argument('-pb','--progressbar',action='store_true',help='display epochs progressbar')
    CLI.add_argument('-co','--cout_rollouts',action='store_true',help='display rollouts cout instead of progressbar')
    args = CLI.parse_args()

    run_imitator(args.cloner,args.envs,args.exp_rollouts,args.nums_epochs,args.progressbar,args.cout_rollouts)
    # main()