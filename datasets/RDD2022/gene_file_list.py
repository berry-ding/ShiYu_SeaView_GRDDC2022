import os
import tqdm
path = 'datasets/RDD2022/India/train/'

files = os.listdir(path+'images/')
for file in tqdm.tqdm(files):
    with open(path+'India.txt', 'a') as f:
        f.write(file[:-4] + '\n')

path = 'datasets/RDD2022/Czech/train/'

files = os.listdir(path+'images/')
for file in tqdm.tqdm(files):
    with open(path+'Czech.txt', 'a') as f:
        f.write(file[:-4] + '\n')

path = 'datasets/RDD2022/Japan/train/'

files = os.listdir(path+'images/')
for file in tqdm.tqdm(files):
    with open(path+'Japan.txt', 'a') as f:
        f.write(file[:-4] + '\n')

path = 'datasets/RDD2022/United_States/train/'

files = os.listdir(path+'images/')
for file in tqdm.tqdm(files):
    with open(path+'United_States.txt', 'a') as f:
        f.write(file[:-4] + '\n')

path = 'datasets/RDD2022/China_Drone/train/'

files = os.listdir(path+'images/')
for file in tqdm.tqdm(files):
    with open(path+'China_Drone.txt', 'a') as f:
        f.write(file[:-4] + '\n')

path = 'datasets/RDD2022/China_MotorBike/train/'

files = os.listdir(path+'images/')
for file in tqdm.tqdm(files):
    with open(path+'China_MotorBike.txt', 'a') as f:
        f.write(file[:-4] + '\n')

path = 'datasets/RDD2022/Norway/train/'

files = os.listdir(path+'images/')
for file in tqdm.tqdm(files):
    with open(path+'Norway.txt', 'a') as f:
        f.write(file[:-4] + '\n')