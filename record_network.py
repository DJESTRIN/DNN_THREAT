#!/usr/bin/env python
""" Custom class that takes network data and saves data to tall format for analysis """
#import dependencies
import torch
import numpy 
import os,glob
import ipdb
import pandas as pd

# Record Neural weights, biases and activity
class Record(object):
  def __init__(self,input_path,env_name):
    self.input_path=input_path
    self.env_name=env_name
      
  def record(self,network,time,episode, network_name):
    filename=self.input_path + network_name + "_" + self.env_name + "_" + str(episode) + "_" + str(time) + ".pt"
    torch.save(network.state_dict(),filename)
    
  def record_activity(self,network_layer_activity, time, episode,network_name,layer_name):
     # function to be placed inside neural network: x=Record.record_activity(nn.relu(nn.linear(x)),time,episode,network_name)
     filename=self.input_path + "/" + network_name + "_" + str(layer_name) + self.env_name + "_" + str(episode) + "_" + str(time) + ".pt"
     torch.save(filename,network_layer_activity)
     return network_layer_activity
  

# Convert torch's pt files to tall format for analysis
class tallformat(object):
  # Takes all recroding data and concatonates it into tall format. 
  def __init__(self,input_path):
    self.input_path=input_path
    self.buildtall()
  
  def get_file_list(self):
    os.chdir(self.input_path)
    return glob.glob('*.pt')
  
  def buildtall(self):
     files=self.get_file_list()
     DF=pd.DataFrame()
     for file in files:
       data=torch.load(file)
       DF_network=pd.DataFrame() 
       for thing in data:
          DF_layer=pd.DataFrame()
          if "weight" in thing:
            data_oh=data[thing]
            if len(data_oh.shape)>3:
              continue
            else:
              data_oh=data_oh.cpu().detach().numpy()
              for i,column in enumerate(data_oh.T):
                ipdb.set_trace()
                neuron_number=np.repeat(i,len(column))
                DF_neuron=pd.DataFrame(neuron_number,column)
                DF_layer=pd.concat(DF_layer,DF_neuron)
          
          if DF_layer.empty:
             continue
          else:
            DF_network=pd.concat(DF_network,DF_layer)
       if DF_network.empty:
          continue
       else:
          DF=pd.concat(DF,DF_network)
              
      
      # Subject, Environment, time, episode, Layer, Neuron, Weight,
        
  
  
