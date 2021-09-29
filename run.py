import os,sys
import time 
start = time.time()
welmsg = str("Welcome to the run.py file, done to run our group's project. Please insert correct numbers, options are displayed and always intergers if not stated otherwise. If you wish to insert your own artwork, make sure it is at minimum about 1040x630 or same dimensions as provided images to run properly with masking mode 0,1,2.")
print(welmsg)
os.system("python " +  os.path.join(sys.path[0], "Editor.py"))
finish = time.time()
print("The process lasted for:", finish-start, "seconds.")
