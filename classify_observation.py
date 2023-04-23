# import subprocess

# # for simple commands
# subprocess.run(["C:\Program Files\ilastik-1.4.0\ilastik.exe --headless --project=BattleZoneVersion0.ilp BattleZoneImageFolder\BattleZone-v4Image33.png BattleZoneImageFolder\BattleZone-v4Image150.png"], shell=True) 

# path = "C:\Program Files\ilastik-1.4.0"

# cd "\Program Files\ilastik-1.3.2"
# $ .\ilastik.exe --headless --project=MyProject.ilp my_next_image1.png my_next_image2.png

# from ilastik.applets.dataSelection.opDataSelection import DatasetInfo
# from ilastik.workflows.pixelClassification import PixelClassificationWorkflow
# import numpy as np

# class classify_observation:
#   def __init__(self,img_array,step,episode):
#     #input into ilastik
#     # Load input data
#     self.datapath = "Battle Zone Image Folder"
#     self.input_data = ilastik.DatasetPath(self.datapath)
#     # Load ilastik project
#     self.ilastikprojectfile = "SpaceInvadersVersion0.ilp"
#     self.ilastik_project = ilastik.ProjectFile(self.ilastikprojectfile)
#     # Load ilastik workflow
#     self.workflow = ilastik.Workflow(self.ilastik_project)
#     # Set input data
#     self.input_slot = self.workflow.inputSlots[0]
#     self.input_slot.setValue(self.input_data)
#   def executefunc(self):
#     # Run ilastik workflow
#     self.workflow.run()
#     # Get the output data
#     self.output_data = self.workflow.outputSlots[0].value
#     print(self.output_data) 
      
# # class BattleZone(classify_observation):
# #   def forward(self):
# #     self.get_common_objects(self)
# #     self.addtional_objects(self)
# #     self.write_data(self)
    
# x = classify_observation()
# x.executefunc()
  








import ilastik
import numpy as np
import subprocess 

class classify_observation(object):
  def __init__(self,img_array,step,episode):
    #input into ilastik
    self.grab_ilastik_output()
    self.get_common_objects()
    self.additiona_objects()
    self.write_data()
    
    def get_common_objects(self):
      #Get agent location
      #Get enemy (number + location)
      
    def write_data(self):
     #save important data for analysis. 
    
    def additiona_objects(self):
      return 
      
class BattleZone(classify_observation):
  def additional_objects(self)
  
  
class SpaceInvaders(classify_observation):
  def additional_objects(self)
  
  
