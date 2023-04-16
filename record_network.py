#!/usr/bin/env python
""" Custom class that takes network data and saves data to tall format for analysis """
#import dependencies
import torch
import numpy 
import os,glob
import ipdb

# Record Neural weights, biases and activity
class Record(object):
  def __init__(self,input_path,env_name):
    self.input_path=input_path
    self.env_name=env_name
      
  def record(self,network,time,episode, network_name):
    ipdb.set_trace()
    os.chdir(self.input_path)
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
     ipdb.set_trace()
     files=self.get_file_list
     for file in files:
       data=torch.load(file)
      
      # Subject, Environment, time, episode, Layer, Neuron, Weight,
        
  
  
