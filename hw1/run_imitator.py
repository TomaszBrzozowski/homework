import argparse
import os
import agents.rl_agent
from util import mkdir_rel
from math import isnan

def run_imitator(cloner,dagger,envs,nums_rollouts,imits_epochs,lrs,bss,*args, **kwargs):
    assert len(envs) == len(nums_rollouts),"argument lists must be the same length"
    assert cloner==True or dagger==True, "must pass at least one imitator agent flag from the following: -c -d"
    inputs = zip(envs,['experts_experience/'+env+'-'+str(nums_rollouts[i])+'_rollouts.pkl' for i,env in enumerate(envs)],
                 nums_rollouts,imits_epochs,lrs if len(lrs)>1 else lrs*len(envs),bss if len(bss)>1 else bss*len(envs))

    n_imit_out_params = 7
    out ={'cloner':[[float('nan')]*n_imit_out_params]*len(envs),'dagger':[[float('nan')]*n_imit_out_params]*len(envs)}
    imitators =[]
    if cloner:
        imitators.append((agents.rl_agent.CloningAgent(),'cloner'))
    if dagger:
        imitators.append((agents.rl_agent.DaggerAgent(),'dagger'))

    def not_nan(tup): return next(k for k in tup if not isnan(k))

    for i, (env_name,expert_data,num_rollouts,num_epochs,lr,batch) in enumerate(inputs):
        for imitator,out_dir in imitators:
            output_dir = os.path.join('imitators_data',env_name,out_dir,''+str(num_rollouts)+'_exp_rollouts')
            mkdir_rel(output_dir)
            ex_mean,ex_std,imit_mean,imit_std,e,s,model_params = imitator.clone_expert(env_name,expert_data,num_epochs,
                                                                                       output_dir,lr,batch,*args,**kwargs)
            out[out_dir][i]=[ex_mean,ex_std,imit_mean,imit_std,e,s,model_params]

        with open('results.txt','a') as f:
            if i==0:
                f.writelines("Model params: "+model_params[:model_params.rindex('_')]+'\n')
            [cl_ex_mean,cl_ex_std,cl_mean,cl_std,e,cl_s,cl_mp],[da_ex_mean,da_ex_std,da_mean,da_std,_,da_s,da_mp] = \
                out['cloner'][i],out['dagger'][i]
            f.writelines(
                "   Env = {0:15s} | Expert: mean(std)={1: >{dm}.{p}f}({2: >{ds}.{p}f}) | "
                "Cloner: mean(std)={3: >{dm}.{p}f}({4: >{ds}.{p}f}), steps={5: >{s}}, {6: >{l}} | "
                "Dagger: mean(std)={7: >{dm}.{p}f}({8: >{ds}.{p}f}), steps={9: >{s}}, {10: >{l}} | exp_rolls={11:6}, batch={12:6}, n_ep={13}\n"
                    .format(envs[i],not_nan((cl_ex_mean,da_ex_mean)),not_nan((cl_ex_std,da_ex_std)),cl_mean,cl_std,cl_s,
                            cl_mp.split('_')[-1],da_mean,da_std,da_s,da_mp.split('_')[-1],num_rollouts,batch,e,p=4,dm=11,ds=11,s=8,l=11))

def run_imitator_all():
    envs_list = ['Ant-v1','HalfCheetah-v1','Hopper-v1','Humanoid-v1','Reacher-v1','Walker2d-v1']
    expert_rollouts_list = [10,10,10,10,10,10]
    epochs_list = [50,50,50,50,50,50]
    run_imitator(True,True,envs_list , expert_rollouts_list,epochs_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Program args
    parser.add_argument('-xt','--xtimes',type=int,default=1,help='run imitator x times')
    parser.add_argument('-c','--cloner',action='store_true',help='run cloner')
    parser.add_argument('-d','--dagger',action='store_true',help='run dagger')
    parser.add_argument('-e','--envnames' ,nargs="*" ,type=str , default=['Ant-v1', ])
    parser.add_argument('-nr','--nums_rollouts' , nargs="*" , type=int ,default=[10, ],help='nums experts r-outs to train on')
    parser.add_argument('-ne','--nums_epochs' , nargs="*" , type=int ,default=[10, ])
    # Model args
    parser.add_argument('-re','--restore', action='store_true')
    parser.add_argument('-l','--layers' , nargs="*" , default=[64], help='num of model hidden layers neurons (d=dropout)')
    parser.add_argument('-nn','--net_name',type=str,default=None)
    parser.add_argument('-kp','--keep_prob',type=float,default=1.0)
    parser.add_argument('-bs','--batch_sizes',nargs="*" ,type=int,default=[64])
    parser.add_argument('-opt','--optimizer',type=str,default='adam',choices=['adam', 'rmsprop', 'adagrad'])
    parser.add_argument('-lr','--lrates',nargs="*" ,type=float,default=[None], help='if None find and use optimal maximum learning rate')
    parser.add_argument('-lrsh','--lr_show',action='store_true',help='display lr finding plot')
    parser.add_argument('-lrs','--lr_schedule',type=str,default='const',choices=['const', 'exp_dec', 'cycle'],
                        help='cycle: inc/dec lr when train i.e cyc=(min->max->min) + tail=(min->min/100)')
    parser.add_argument('-lreb','--lr_exp_base',type=int,default=100)
    parser.add_argument('-lred','--lr_exp_decay',type=float,default=0.99)
    parser.add_argument('-lrmm','--lr_maxmin',type=int,default=10,help='lr cycle max/min ratio')
    parser.add_argument('-lrct','--lr_c_t_ratio',type=int,default=9, help='ratio of cycle/tail duration')
    # Rl agent args
    parser.add_argument('-ntr','--num_test_rollouts' , type=int ,default=10, help='num r-outs testing imitator policy')
    parser.add_argument('-mt','--max_steps' , type=int ,default=None,help='max timesteps for test rollouts')
    parser.add_argument('-pb','--progbar',action='store_true',help='display epochs progressbar')
    parser.add_argument('-r','--render', action='store_true', help='render expert/imitator rollouts')
    parser.add_argument('-co','--cout_r',action='store_true',help='display rollouts progress printed instead of progressbar')
    parser.add_argument('-rec','--record',action='store_true',help='record rollouts videos')
    args = parser.parse_args()

    for i in range(args.xtimes):
        run_imitator(args.cloner,args.dagger,args.envnames,args.nums_rollouts,args.nums_epochs,args.lrates,args.batch_sizes,
                     args.restore,args.layers,args.net_name,args.keep_prob,args.optimizer,args.lr_show,args.lr_schedule,
                     args.lr_exp_base,args.lr_exp_decay,args.lr_maxmin,args.lr_c_t_ratio,args.num_test_rollouts,
                     args.max_steps,args.progbar,args.render,args.cout_r,args.record)