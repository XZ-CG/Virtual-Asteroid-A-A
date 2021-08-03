# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 19:39:39 2019

@author: lxz
"""
import os
import subprocess
import datetime
import numpy as np
import re

output_path = os.path.abspath(os.path.join(os.path.dirname("__file__"),os.path.pardir))+"/asteroid/point1/Assets"+"/"

#----------------------Launch the exe file (execute on GPU side)----------------------------
def compute(i): 
    name = "point%s"%i
    path = os.getcwd()
    file = path + "\\Asteroid.exe"
    subprocess.call([file,name])
    return 1.0    

if __name__ == "__main__":
    start_time = datetime.datetime.now().time().strftime('%H:%M:%S')
    # create_h_file("Bennu.spheres", metaball=True)
    # create_h_file("output.spheres", metaball=True)
    
    print(compute(1))
    
    end_time = datetime.datetime.now().time().strftime('%H:%M:%S')
    total_time=(datetime.datetime.strptime(end_time,'%H:%M:%S') - datetime.datetime.strptime(start_time,'%H:%M:%S'))
    print(total_time)







