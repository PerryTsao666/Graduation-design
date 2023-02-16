import cv2
import glob
import re
import os


list = glob.glob('raw-890/' + '//*.png')

index1 = []
for i in range(len(list)):
    if len(list[i].split('_')) == 3:
        index = int(re.split(r"\\", list[i].split('_')[0])[1])
    else:
        index = int((re.split(r"\\", list[i].split('_')[0])[1]).split('.')[0])
    index1.append(index)
    img = cv2.resize(cv2.imread(list[i]), (640, 480))
    cv2.imwrite('./train/{}.png'.format(index), img)
    if os.path.isfile('./train/{}.png'.format(index)):
        pass
    else:
        print(index)
        break
print(len(index1))
print('done')
