import argparse
import os
import agents.rl_agent
from util import mkdir_rel
from math import isnan

def run_imitator(cloner,dagger,envs,nums_rollouts,imits_epochs,layers=[64,0],net_name=None,batch=64,drop_prob=0.5,
                 render=False,pbar=False,cout=False,record=False):
    assert len(envs) == len(nums_rollouts),"argument lists must be the same length"
    assert cloner==True or dagger==True, "must pass at least one imitator agent flag from the following: -c -d"
    inputs = zip(envs,['experts_experience/'+env+'-'+str(nums_rollouts[i])+'_rollouts.pkl' for i,env in enumerate(envs)],nums_rollouts,imits_epochs)
    if net_name==None: net_name = '_'.join(str(e) for e in layers)

    out ={'cloner':[[float('nan')]*6]*len(envs),'dagger':[[float('nan')]*6]*len(envs)}
    imitators =[]
    if cloner:
        imitators.append((agents.rl_agent.CloningAgent(),'cloner'))
    if dagger:
        imitators.append((agents.rl_agent.DaggerAgent(),'dagger'))

    for i, (env_name,expert_data,num_rollouts,num_epochs) in enumerate(inputs):
        for imitator,res_dir in imitators:
            output_dir = os.path.join('imitators_data',env_name,res_dir,''+str(num_rollouts)+'_exp_rollouts','nn'+net_name)
            mkdir_rel(output_dir)
            ex_mean,ex_std,imit_mean,imit_std, _, step = imitator.clone_expert(env_name,expert_data,output_dir,layers,batch,
                                                                        drop_prob,num_epochs=num_epochs,num_test_rollouts=1,
                                                                        restore=True,render=render,progbar=pbar,cout=cout,record=record)
            out[res_dir][i]=[ex_mean,ex_std,imit_mean,imit_std,step]

    def not_nan(tup): return next(k for k in tup if not isnan(k))

    with open('results.txt','a') as f:
        for i, ([cl_ex_mean, cl_ex_std, cl_mean, cl_std, st], [da_ex_mean, da_ex_std, da_mean, da_std,_]) in enumerate(zip(out['cloner'],out['dagger'])):
            f.writelines("Env = {0} | Expert: mean(std)={1:.{p}f}({2:.{p}f}) | Cloner: mean(std)={3:.{p}f}({4:.{p}f})"
                         " | Dagger: mean(std)={5:.{p}f}({6:.{p}f}) | exp_rolls={8}, nn_layers={7}, batch={8}, drop_prob={9},"
                         " n_epochs={10}, n_steps={10} \n".format(envs[i], not_nan((cl_ex_mean, da_ex_mean)), not_nan((cl_ex_std, da_ex_std))
            ,cl_mean, cl_std, da_mean, da_std, nums_rollouts[i], layers, batch, drop_prob, imits_epochs[i], st, p=4))

def run_imitator_all():
    envs_list = ['Ant-v1','HalfCheetah-v1','Hopper-v1','Humanoid-v1','Reacher-v1','Walker2d-v1']
    expert_rollouts_list = [10,10,10,10,10,10]
    epochs_list = [50,50,50,50,50,50]
    run_imitator(True,True,envs_list , expert_rollouts_list,epochs_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--cloner',action='store_true',help='run cloner')
    parser.add_argument('-d','--dagger',action='store_true',help='run dagger')
    parser.add_argument('-e','--envnames' ,nargs="*" ,type=str , default=['Ant-v1', ])
    parser.add_argument('-nr','--nums_rollouts' , nargs="*" , type=int ,default=[10, ])
    parser.add_argument('-ne','--nums_epochs' , nargs="*" , type=int ,default=[50, ])
    parser.add_argument('-l','--layers' , nargs="*" , type=int ,default=[64,0], help='num of model layers neurons (0=dropout)')
    parser.add_argument('-nn','--net_name',type=str,default=None)
    parser.add_argument('-bs','--batch_size',type=int,default=64)
    parser.add_argument('-dp','--drop_prob',type=float,default=0.5)
    parser.add_argument('-r','--render', action='store_true')
    parser.add_argument('-pb','--progressbar',action='store_true',help='display epochs progressbar')
    parser.add_argument('-co','--cout_rollouts',action='store_true',help='display rollouts progress printed instead of progressbar')
    parser.add_argument('-rec','--record',action='store_true',help='record rollouts videos')
    args = parser.parse_args()

    run_imitator(args.cloner,args.dagger,args.envnames,args.nums_rollouts,args.nums_epochs,args.layers,args.net_name,
                 args.batch_size,args.drop_prob,args.render,args.progressbar,args.cout_rollouts,args.record)