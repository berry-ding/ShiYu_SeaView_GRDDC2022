import os
import errno
import pwd
# for country in ["China_Drone", "China_MotorBike", "Czech", "India", "Norway", "United_States"]:
#     pass
countries = os.listdir()
# import pdb;pdb.set_trace()
for country in countries:
    if country not in ["China_Drone", "China_MotorBike", "Czech", "India", "Norway", "United_States", 'Japan']:
        continue
    images_path = os.path.join(country, 'train', 'images')
    image_list = os.listdir(images_path)
    with open('train.txt', 'a') as f:
        for img in image_list:
            img_path = os.path.join('datasets/RDD2022/', images_path, img)
            if img != 'Czech_002179.jpg':
                f.write(img_path + '\n')
with open('val.txt', 'a') as f:
    images_path = 'Czech/train/images/'
    img = 'Czech_002179.jpg'
    img_path = os.path.join('datasets/RDD2022/', images_path, img)
    f.write(img_path + '\n')
        