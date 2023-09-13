#!/usr/bin/env python3

""" Custom class that takes network data and saves data to tall format for analysis """
#import dependencies
import torch
import numpy 
import os,glob
import ipdb
import pandas as pd
import numpy as np
import csv

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
        # following line was missing an underscore after layer name
        filename=self.input_path + "/" + network_name + "_" + str(layer_name) + "_" + self.env_name + "_" + str(episode) + "_" + str(time) + ".pt"
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
    
    def get_file_info(self, file):
        #to get info from the filename
        filename = os.path.splitext(file)[0]
        info_list = filename.split('_')
        network_name = info_list[0]
        ## layer_name_s = info_list[?]
        env_name = info_list[1]   
        episode_name_s = info_list[2]
        time_name_s = info_list[3]
        return network_name, env_name, episode_name_s, time_name_s
  
    def buildtall(self):
        files=self.get_file_list()
        
        with open('output.csv', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            # writing the header row
            csv_writer.writerow(["Network Name", "Environment Name", "Episode Name", "Time Name"])
            
            for file in files:
                network_name, env_name, episode_name_s, time_name_s = self.get_file_info(file)
                # writing each row in the CSV file
                csv_writer.writerow([network_name, env_name, episode_name_s, time_name_s])

if __name__ == "__main__":
    input_path = "/Users/dhritimamtora/"
    tall_format_processor = tallformat(input_path)


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
