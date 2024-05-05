from distutils.log import warn
import torch
import numpy as np
import torch
import time
import time
import ray

def Build_Mapping_Matrix(models):
    """
    Given a list of COBRA model objects, this function will build a mapping matrix for all the exchange reactions

    """

    Ex_sp = []
    Ex_rxns = []
    for model in models:
        Ex_rxns.extend([(model,list(model.reactions[rxn].metabolites)[0].id,rxn) for rxn in model.exchange_reactions if model.reactions[rxn].id.endswith("_e") and rxn!=model.biomass_ind])
    Ex_sp=list(set([item[1] for item in Ex_rxns]))
    Mapping_Matrix = np.full((len(Ex_sp), len(models)),-1, dtype=int)
    for record in Ex_rxns:
        Mapping_Matrix[Ex_sp.index(record[1]),models.index(record[0])]=record[2]

    return {"Ex_sp": Ex_sp, "Mapping_Matrix": Mapping_Matrix}

def general_kinetic(x,y):
    """
    A simple function implementing MM kinetics
    """
    return 0.1*x*y/(10+x)
def general_uptake(c):
    """
    An extremely simple function for mass transfer kinetic
    """
    return 10*(c/(c+10))

def mass_transfer(x,y,k=0.01):
    """
    A simple function for mass transfer kinetic
    """
    return k*(x-y)

@ray.remote
def run_episode(env):
    """ Runs a single episode of the environment used for parallel computatuon of episodes
    """
    t_0_ep=time.time()
    batch_obs = {key.name:[] for key in env.agents}
    batch_acts = {key.name:[] for key in env.agents}
    batch_log_probs = {key.name:[] for key in env.agents}
    episode_rews = {key.name:[] for key in env.agents}
    env.reset()
    episode_len=env.episode_length
    for ep in range(episode_len):
        env.t=episode_len-ep
        obs = env.state.copy()
        for agent in env.agents:
            action, log_prob = agent.get_actions(np.hstack([obs[agent.observables],env.t]))
            agent.a=action
            agent.log_prob=log_prob.detach()
        t_0_step=time.time()
        s,r,a,sp=env.step()
        env.time_dict["step"].append(time.time()-t_0_step)
        for ind,ag in enumerate(env.agents):
            batch_obs[ag.name].append(np.hstack([s[ag.observables],env.t]))
            batch_acts[ag.name].append(a[ind])
            batch_log_probs[ag.name].append(ag.log_prob)
            episode_rews[ag.name].append(r[ind])
        env.time_dict["step"].append(time.time()-t_0_step)
    env.time_dict["episode"].append(time.time()-t_0_ep)
    return batch_obs,batch_acts, batch_log_probs, episode_rews,env.time_dict,env.rewards

def run_episode_single(env):
    """
    Runs a single episode of the environment
    """
    batch_obs = {key.name:[] for key in env.agents}
    batch_acts = {key.name:[] for key in env.agents}
    batch_log_probs = {key.name:[] for key in env.agents}
    episode_rews = {key.name:[] for key in env.agents}
    env.reset()
    episode_len=env.episode_length
    for ep in range(episode_len):
        env.t=episode_len-ep
        obs = env.state.copy()
        for agent in env.agents:
            action, log_prob = agent.get_actions(np.hstack([obs[agent.observables],env.t]))
            agent.a=action
            agent.log_prob=log_prob .detach()
        s,r,a,sp=env.step()
        for ind,ag in enumerate(env.agents):
            batch_obs[ag.name].append(np.hstack([s[ag.observables],env.t]))
            batch_acts[ag.name].append(a[ind])
            batch_log_probs[ag.name].append(ag.log_prob)
            episode_rews[ag.name].append(r[ind])
    return batch_obs,batch_acts, batch_log_probs, episode_rews

def rollout(env,num_workers=None):
    """
    Performs a batch calculation in parallel using Ray library
    Args:
        env (Environment): The environment instance to run the episodes for
    """
    if num_workers is None:
        num_workers=env.episodes_per_batch
    t0_batch=time.time()
    batch_obs={key.name:[] for key in env.agents}
    batch_acts={key.name:[] for key in env.agents}
    batch_log_probs={key.name:[] for key in env.agents}
    batch_rews = {key.name:[] for key in env.agents}
    batch_rtgs = {key.name:[] for key in env.agents}
    batch_times={"step":[], "episode":[], "optimization":[], "batch":[]}
    batch=[]
    env.reset()

    for ep in range(num_workers):
        batch.append(run_episode.remote(env))
    batch=ray.get(batch)
    for ep in range(num_workers):
        for ag in env.agents:
            batch_obs[ag.name].extend(batch[ep][0][ag.name])
            batch_acts[ag.name].extend(batch[ep][1][ag.name])
            batch_log_probs[ag.name].extend(batch[ep][2][ag.name])
            batch_rews[ag.name].append(batch[ep][3][ag.name])
        batch_times["step"].extend(batch[ep][4]["step"])
        batch_times["episode"].extend(batch[ep][4]["episode"])
        batch_times["optimization"].extend(batch[ep][4]["optimization"])

    for ag in env.agents:
        env.rewards[ag.name].extend(list(np.sum(np.array(batch_rews[ag.name]),axis=1)))

    for agent in env.agents:

        batch_obs[agent.name] = torch.tensor(batch_obs[agent.name], dtype=torch.float)
        batch_acts[agent.name] = torch.tensor(batch_acts[agent.name], dtype=torch.float)
        batch_log_probs[agent.name] = torch.tensor(batch_log_probs[agent.name], dtype=torch.float)
        batch_rtgs[agent.name] = agent.compute_rtgs(batch_rews[agent.name])
    batch_times["batch"].append(time.time()-t0_batch)
    return batch_obs,batch_acts, batch_log_probs, batch_rtgs,batch_times,env.rewards.copy()
 
"""
The code is completely inspired from
https://github.com/chan-csu/SPAM-DFBA.git
"""