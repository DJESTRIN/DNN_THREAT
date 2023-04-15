import torch
import numpy 
import glob

""" Custom class that takes network and saves data to common file """
class Record:
  def __init__(self,input_path):
    self.input_path=input_path
      
  def record(self,network,time,episode):
  
  
class tallformat:
  # Takes all recroding data and concatonates it into tall format. 
  def __init__(self,input_path):
    self.input_path=input_path
  
  def get_file_list(self):
    glob.glob(
  
