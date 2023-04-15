#!/usr/bin/env python
""" Custom class that takes network data and saves data to tall format for analysis """
#import dependencies
import torch
import numpy 
import os,glob

#
class Record:
  def __init__(self,input_path):
    self.input_path=input_path
      
  def record(self,network,env_name,time,episode):
    os.chdir(self.input_path)
    filename=env_name + "_" + str(episode) + "_" + time + ".pt"
    torch.save(filename,network.state_dict)
  
  
class tallformat:
  # Takes all recroding data and concatonates it into tall format. 
  def __init__(self,input_path):
    self.input_path=input_path
    self.buildtall()
  
  def get_file_list(self):
    os.chdir(self.input_path)
    return glob.glob('*.pt')
  
  def buildtall(self)
      files=self.get_file_list
      for file in files:
        data=torch.load(file)
        
  
  
