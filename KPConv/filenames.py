# import OS module
import os
import random
# Get the list of all files and directories
path = "/home/reza/PHD/Data/olivier/plyfiles"


for subplot in (os.listdir(path)):
    subplot_path = os.path.join(path, subplot)
    for plot in (os.listdir(subplot_path)):
        new_cloud = subplot + "_" + plot + "_points.ply"
        new_name = os.path.join("/home/reza/PHD/Data/olivier/plyfiles/", new_cloud)
        plot_path = os.path.join(subplot_path, plot)
        cloud_path = os.path.join(plot_path, "points.ply")
        os.rename(cloud_path, new_name)       


all_clouds = []
for plot in (os.listdir(path)):
    plot_path = os.path.join(path, plot)
    all_clouds.append(plot_path)
    

random.shuffle(all_clouds)
train_split = int(7*len(all_clouds)/10)
validation_split = int(8*len(all_clouds)/10)

train_filenames = all_clouds[:train_split]
valid_filenames = all_clouds[train_split:validation_split]
test_filenames = all_clouds[validation_split:]
# valid_filenames1 = []

with open('train.txt', 'w') as f:
    for line in train_filenames:
        f.write(f"{line}\n")