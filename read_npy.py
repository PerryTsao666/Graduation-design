import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import torchvision.transforms as transforms
import cv2
import PIL.Image as Image
import numpy as np


class triplet_train(Dataset):
    def __init__(self):
        path1 = './train/'
        path2 = './val'
        self.list = glob.glob(path1 + '/*.png')
        self.list2 = glob.glob(path2 + '/*.png')
        self.list.extend(self.list2)
        self.size = len(self.list)
        print(self.size)
        self.transform_a = transforms.Compose([transforms.Resize([256, 256]),
                                               transforms.ToTensor()])

    def __getitem__(self, index):
        image_id = int(self.list[index].split('.')[1].split('/')[2])
        xt0 = Image.open(self.list[index])
        xt0 = torch.from_numpy(np.array(self.transform_a(xt0)))

        return xt0, image_id

    def __len__(self):
        return self.size


if __name__ == '__main__':
    data = triplet_train()
    data_load = DataLoader(data, batch_size=1, num_workers=4)
    data_iter = data_load.__iter__()

    model = torchvision.models.resnet101(pretrained=True)
    backbone = torch.nn.Sequential(*list(model.children())[:-2])

    feature = []
    ids = []

    for i in range(len(data_load)):
        img, image_id = data_iter.next()
        out = backbone(img).view(1, 2048, -1).squeeze(0).permute(1, 0).detach().numpy()
        feature.append(out)
        ids.append(image_id.item())
    npy_file = dict(zip(ids, feature))
    print(type(npy_file))
    np.save('train.npy', npy_file)

    a = np.load('train.npy', allow_pickle=True)
    print(a.item().get(400))
