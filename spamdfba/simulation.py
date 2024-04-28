import torch
import numpy as np
import torch
import torch.nn as nn
import pickle
import time
import pandas as pd
import os
import time
import plotly.graph_objs as go
from rich.console import Console
from rich.table import Table
from mapping_matrix import rollout

DEFAULT_PLOTLY_COLORS=['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                       'rgb(148, 103, 189)', 'rgb(140, 86, 75)', 'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                       'rgb(188, 189, 34)', 'rgb(23, 190, 207)']*10
DEFAULT_PLOTLY_COLORS_BACK=['rgba(31, 119, 180,0.2)', 'rgba(255, 127, 14,0.2)',
                       'rgba(44, 160, 44,0.2)', 'rgba(214, 39, 40,0.2)',
                       'rgba(148, 103, 189,0.2)', 'rgba(140, 86, 75,0.2)',
                       'rgba(227, 119, 194,0.2)', 'rgba(127, 127, 127,0.2)',
                       'rgba(188, 189, 34,0.2)', 'rgba(23, 190, 207,0.2)']*10

class Simulation:
    """
    This class is designed to run the final simulation for an environment and additionaly does
    -> Saving the results given a specific interval
    -> Plotting the results
    -> calculating the duration of different parts of the code

    Args:
        name (str): A descriptive name given to the simulation. This name is used to save the training files
        env (environment): The environment to perform the simulations in
        save_dir (str): The DIRECTORY to which you want to save the training results
        overwrite (bool): Determines whether to overwrite the pickel in each saving interval create new files
        report (dict): Includes the reported time at each step
    """

    def __init__(self,name,env,save_dir,store_return, save_every=200,overwrite=False):
        self.name=name
        self.env=env
        self.save_dir=save_dir
        self.store_return = store_return
        self.save_every=save_every
        self.overwrite=overwrite
        self.report={}


    def run(self,solver="glpk",verbose=True,initial_critic_error=100):
        """
        This method runs the training loop

        Args:
            solver (str): The solver to be used by cobrapy
            verbose (bool): whether to print the training results after each iteration
            initial_critic_error (float): To make the training faster this method first trains the critic network on the 
            first batch of episodes to make the critic network produce more realistic values in the beginning. This parameter 
            defines what is the allowable MSE of the critic network on the first batch of data obtained from the evironment
        Returns:
            Environment: The trained version of the environment
        """
        
        t_0_sim=time.time()
        self.report={"returns":{ag.name:[] for ag in self.env.agents}}
        self.report["times"]={
            "step":[], "optimization":[], "batch":[], "simulation":[]}
        if not os.path.exists(os.path.join(self.save_dir,self.name)):
            os.makedirs(os.path.join(self.save_dir,self.name))

        for agent in self.env.agents:
            agent.model.solver=solver

        for batch in range(self.env.number_of_batches):
            batch_obs,batch_acts, batch_log_probs, batch_rtgs,batch_times,env_rew=rollout(self.env)
            self.report["times"]["step"].append(np.mean(batch_times["step"]))
            self.report["times"]["optimization"].append(np.mean(batch_times["optimization"]))
            self.report["times"]["batch"].append(np.mean(batch_times["batch"]))
            for agent in self.env.agents:
                self.report["returns"][agent.name].append(env_rew[agent.name])
                V, _= agent.evaluate(batch_obs[agent.name],batch_acts[agent.name])
                A_k = batch_rtgs[agent.name] - V.detach()
                A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-5)
                if batch==0:
                    if verbose:
                        print("Hold on, bringing the creitc network to range ...")
                        err=initial_critic_error+1
                        while err>initial_critic_error:
                            V, _= agent.evaluate(batch_obs[agent.name],batch_acts[agent.name])
                            critic_loss = nn.MSELoss()(V, batch_rtgs[agent.name])
                            agent.optimizer_value_.zero_grad()
                            critic_loss.backward()
                            agent.optimizer_value_.step()
                            err=critic_loss.item()
                    if verbose:
                        print("Done!")
                else:
                    for _ in range(agent.grad_updates):

                        V, curr_log_probs = agent.evaluate(batch_obs[agent.name],batch_acts[agent.name])
                        ratios = torch.exp(curr_log_probs - batch_log_probs[agent.name])
                        surr1 = ratios * A_k.detach()
                        surr2 = torch.clamp(ratios, 1 - agent.clip, 1 + agent.clip) * A_k
                        actor_loss = (-torch.min(surr1, surr2)).mean()
                        critic_loss = nn.MSELoss()(V, batch_rtgs[agent.name])
                        agent.optimizer_policy_.zero_grad()
                        actor_loss.backward(retain_graph=False)
                        agent.optimizer_policy_.step()
                        agent.optimizer_value_.zero_grad()
                        critic_loss.backward()
                        agent.optimizer_value_.step()

                if batch%self.save_every==0:
                    if self.overwrite:
                        with open(os.path.join(self.save_dir,self.name,self.name+".pkl"), 'wb') as f:
                            pickle.dump(self.env, f)
                        with open(os.path.join(self.save_dir,self.name,self.name+"_obs.pkl"), 'wb') as f:
                            pickle.dump(batch_obs,f)
                        with open(os.path.join(self.save_dir,self.name,self.name+"_acts.pkl"), 'wb') as f:
                            pickle.dump(batch_acts,f)
                    else:
                        with open(os.path.join(self.save_dir,self.name,self.name+f"_{batch}"+".pkl"), 'wb') as f:
                            pickle.dump(self.env, f)
                        with open(os.path.join(self.save_dir,self.name,self.name+f"_{batch}"+"_obs.pkl"), 'wb') as f:
                            pickle.dump(batch_obs,f)
                        with open(os.path.join(self.save_dir,self.name,self.name+f"_{batch}"+"_acts.pkl"), 'wb') as f:
                            pickle.dump(batch_acts,f)

            if verbose:
                print(f"Batch {batch} finished:")
                for agent in self.env.agents:
                    print(f"{agent.name} return was:  {np.mean(self.env.rewards[agent.name][-self.env.episodes_per_batch:])}")
                    self.store_return.append(np.mean(self.env.rewards[agent.name][-self.env.episodes_per_batch:]))
                    
        self.report["times"]["simulation"].append(time.time()-t_0_sim)

    def plot_learning_curves(self,plot=True):
        """
        This method plots the learning curve for all the agents
        Args:
            plot (bool): whether to render the plot as well

        Returns:
            go.Figure : Returns a plotly figure for learning curves of the agents
        """
        fig = go.Figure()
        for index,agent in enumerate(self.env.agents):
            rets=pd.DataFrame(self.report["returns"][agent.name])
            x=rets.index.to_list()
            fig.add_trace(go.Scatter(
                x=x,
                y=rets.mean(axis=1).to_list(),
                line=dict(color=DEFAULT_PLOTLY_COLORS[index]),
                name=agent.name,
                mode='lines'
                        ))
            fig.add_trace(go.Scatter(
                        x=x+x[::-1],
                        y=rets.max(axis=1).to_list()+rets.min(axis=1).to_list()[::-1],
                        fill='toself',
                        fillcolor=DEFAULT_PLOTLY_COLORS_BACK[index],
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=False)
                            )
            fig.update_layout(
                xaxis={
                    "title":"Batch"
                },
                yaxis={
                    "title":"Total Episode Return"
                }

            )
        if plot:
            fig.show()
        return fig

    def print_training_times(self,draw_table=True):
        """
        Returns a dictionary describing the simulation time at different level of the training process
        You can also opt to draw a table based on this results using Rich library

        Args:
            draw_table (bool): whether to draw the table in the console

        Returns:
            dict: A list of dictionaries that contain duration of execution for different stages of simulation
        """
        report_times=pd.concat([pd.DataFrame.from_dict(self.report["times"],orient='index').fillna(method="ffill",axis=1).mean(axis=1),pd.DataFrame.from_dict(self.report["times"],orient='index').fillna(method="ffill",axis=1).std(axis=1)],axis=1).rename({0:"mean",1:"std"},axis=1).to_dict(orient='index')
        if draw_table:
            table = Table(title="Simulation times")
            table.add_column("Level", justify="left", style="cyan", no_wrap=True)
            table.add_column("Mean(s)", style="cyan",justify="left")
            table.add_column("STD(s)", justify="left", style="cyan")
            table.add_row("Optimization",str(report_times[1]["mean"]),str(report_times[1]["std"]))
            table.add_row("Step",str(report_times[0]["mean"]),str(report_times[0]["std"]))
            table.add_row("Batch",str(report_times[2]["mean"]),str(report_times[2]["std"]))
            table.add_row("Simulation",str(report_times[3]["mean"]),"NA")
            console = Console()
            console.print(table)
        return report_times
    
    
"""
The code is completely inspired from
https://github.com/chan-csu/SPAM-DFBA.git
"""