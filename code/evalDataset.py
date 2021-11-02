import paddle
from paddle.io import Dataset
from PIL import Image
import os

class EvalDataset(Dataset):
    def __init__(self, data_dir):
        super(EvalDataset, self).__init__()
        img_list = os.listdir(data_dir)
        self.img_list = []
        for item in img_list:
            if item[0]=='.':
                continue
            self.img_list.append(item)
        # print(self.img_list)
        self.data_dir = data_dir

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.img_list[index])
        img = Image.open(img_path)
        img = paddle.vision.transforms.to_tensor(img)
        return img
    
    def __len__(self):
        return len(self.img_list)