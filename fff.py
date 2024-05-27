import os
from os import listdir
 
# get the path/directory
folder_dir = "./dataset/frontminor"
for images in os.listdir(folder_dir):
    print(images)

len(os.listdir(folder_dir))
