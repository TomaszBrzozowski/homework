import argparse
import os
import agents.rl_agent
from util import mkdir_rel
from math import isnan

def run_imitator(cloner,dagger,envs,nums_rollouts,imits_epochs,render=False,pbar=False,cout=False,record=False):
    assert len(envs) == len(nums_rollouts),"argument lists must be the same length"
    assert cloner==True or dagger==True, "must pass at least one imitator agent flag from the following: -c -d"
    inputs = zip(envs,['experts_experience/'+env+'-'+str(nums_rollouts[i])+'_rollouts.pkl' for i,env in enumerate(envs)],nums_rollouts,imits_epochs)

    results ={'cloner':[[float('nan')]*6]*len(envs),'dagger':[[float('nan')]*6]*len(envs)}
    imitators =[]
    if cloner:
        imitators.append((agents.rl_agent.CloningAgent(),'cloner'))
    if dagger:
        imitators.append((agents.rl_agent.DaggerAgent(),'dagger'))

    for env_name,expert_data,num_rollouts,num_epochs in inputs:
        for imitator,res_dir in imitators:
            results[res_dir].clear()
            output_dir = os.path.join('imitators_data',env_name,res_dir,''+str(num_rollouts)+'_exp_rollouts')
            mkdir_rel(output_dir)
            ex_mean,ex_std,imit_mean,imit_std, epoch = imitator.clone_expert(env_name,expert_data,model_output_dir=output_dir,
                                                                        restore=True,render=render,num_epochs=num_epochs,
                                                                        num_test_rollouts=2,progbar=pbar,cout=cout,record=record)
            results[res_dir].append([ex_mean,ex_std,imit_mean,imit_std,num_rollouts,epoch])

    with open('results.txt','a') as f:
        for i, ([cl_ex_mean, cl_ex_std, cl_mean, cl_std, cl_nr, cl_ne], [da_ex_mean, da_ex_std, da_mean, da_std, da_nr, da_ne]) \
                in enumerate(zip(results['cloner'],results['dagger'])):
            f.writelines("Env = {0} | Expert: mean(std)={1:.{p}f}({2:.{p}f}) | Cloner: mean(std)={3:.{p}f}({4:.{p}f}), exp_rolls={5}, n_epochs={6}"
                         " | Dagger: mean(std)={7:.{p}f}({8:.{p}f}), exp_rolls={9}, n_epochs={10}\n"
                         .format(envs[i], next(k for k in [cl_ex_mean, da_ex_mean] if not isnan(k)), next(k for k in [cl_ex_std, da_ex_std] if not isnan(k))
            ,cl_mean, cl_std, cl_nr, cl_ne, da_mean, da_std, da_nr, da_ne,p=4))

def run_imitator_all():
    envs_list = ['Ant-v1','HalfCheetah-v1','Hopper-v1','Humanoid-v1','Reacher-v1','Walker2d-v1']
    expert_rollouts_list = [10,10,10,10,10,10]
    epochs_list = [50,50,50,50,50,50]
    run_imitator(True,True,envs_list , expert_rollouts_list,epochs_list,False,False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--cloner',action='store_true',help='run cloner')
    parser.add_argument('-d','--dagger',action='store_true',help='run dagger')
    parser.add_argument('-e','--envnames' ,nargs="*" ,type=str , default=['Ant-v1', ])
    parser.add_argument('-nr','--nums_rollouts' , nargs="*" , type=int ,default=[10, ])
    parser.add_argument('-ne','--nums_epochs' , nargs="*" , type=int ,default=[50, ])
    parser.add_argument('-r','--render', action='store_true')
    parser.add_argument('-pb','--progressbar',action='store_true',help='display epochs progressbar')
    parser.add_argument('-co','--cout_rollouts',action='store_true',help='display rollouts progress printed instead of progressbar')
    parser.add_argument('-rec','--record',action='store_true',help='record rollouts videos')
    args = parser.parse_args()

    run_imitator(args.cloner,args.dagger,args.envnames,args.nums_rollouts,args.nums_epochs,args.render,args.progressbar,args.cout_rollouts,args.record)