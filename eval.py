import argparse
import json
import os
from tqdm import tqdm
import io

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from models.sinet import SiNet
from models.slinet import SliNet
from utils.datautils.core50data import CORE50

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorithms.')
    parser.add_argument('--scenario', type=str, default='cddb_hard', help='scenario to test')
    parser.add_argument('--resume', type=str, default='', help='resume model')
    parser.add_argument('--compression', type=bool, default=False, help='test on compressed data')
    parser.add_argument('--config', type=str, default='configs/cddb_slip.json', help='Json file of settings.')
    parser.add_argument('--dataroot', type=str, default='/home/francesco.laiti/datasets/CDDB/', help='data path')
    parser.add_argument('--datatype', type=str, default='deepfake', help='data type')
    parser.add_argument('--random_select', action='store_true', help='use random select')
    parser.add_argument('--til', action='store_true', help='use groundtruth task identification')
    return parser

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param

class DummyDataset(Dataset):
    def __init__(self, data_path, data_type, data_scenario, data_compression):
        self.do_compress = [data_compression, 50] # enable/disable compression from flag - jpeg quality
        self.trsf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        images = []
        labels = []
        print(f'--- Data compression enabled: {self.do_compress} ---')
        if data_type == "deepfake":
            if data_scenario == "cddb_hard":
                subsets = ["gaugan", "biggan", "wild", "whichfaceisreal", "san"] # <- CDDB Hard
                multiclass = [0,0,0,0,0]
            elif data_scenario == "ood":
                subsets = ["deepfake", "glow", "stargan_gf"] # <- OOD experiments
                multiclass = [0,1,1]
            else:
                raise RuntimeError(f"Unexpected data_scenario value: {data_scenario}. Expected 'cddb_hard' or 'ood'.")
            print(f'--- Test on {subsets} with {data_scenario} scenario ---')
            for id, name in enumerate(subsets):
                root_ = os.path.join(data_path, name, 'val')
                # sub_classes = ['']
                sub_classes = os.listdir(root_) if multiclass[id] else ['']
                for cls in sub_classes:
                    for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                        images.append(os.path.join(root_, cls, '0_real', imgname))
                        labels.append(0 + 2 * id)

                    for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                        images.append(os.path.join(root_, cls, '1_fake', imgname))
                        labels.append(1 + 2 * id)
        elif data_type == "domainnet":
            self.data_root = data_path
            self.image_list_root = self.data_root
            self.domain_names = ["clipart","infograph","painting","quickdraw", "real","sketch",]
            image_list_paths = [os.path.join(self.image_list_root, d + "_" + "test" + ".txt") for d in self.domain_names]
            imgs = []
            for taskid, image_list_path in enumerate(image_list_paths):
                image_list = open(image_list_path).readlines()
                imgs += [(val.split()[0], int(val.split()[1])+taskid*345) for val in image_list]

            for item in imgs:
                images.append(os.path.join(self.data_root, item[0]))
                labels.append(item[1])
        elif data_type == "core50":
            self.dataset_generator = CORE50(root=data_path, scenario="ni")
            images, labels = self.dataset_generator.get_test_set()
            labels = labels.tolist()
        else:
            pass

        assert len(images) == len(labels), 'Data size error!'
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.trsf(self.pil_loader(self.images[idx], self.do_compress[0], self.do_compress[1]))
        label = self.labels[idx]
        return idx, image, label

    def pil_loader(self, path, do_compress, quality):
        with open(path, 'rb') as f:
            if do_compress:
                f = self.compress_image_to_memory(path, quality=quality)
            img = Image.open(f)
            return img.convert('RGB')
    
    def compress_image_to_memory(self, path, quality):
        with Image.open(path) as img:
            output = io.BytesIO()
            img.save(output, "JPEG", quality=quality)
            output.seek(0)
            return output

def accuracy_domain(y_pred, y_true, increment=10):
    assert len(y_pred) == len(y_true), 'Data length error.'
    all_acc = {}
    all_acc['total'] = np.around((y_pred%345 == y_true%345).sum()*100 / len(y_true), decimals=2)
    return all_acc

def accuracy_core50(y_pred, y_true):
    assert len(y_pred) == len(y_true), 'Data length error.'
    all_acc = {}
    all_acc['total'] = np.around((y_pred == y_true).sum()*100 / len(y_true), decimals=2)
    return all_acc

def accuracy_binary(y_pred, y_true, increment=2):
    assert len(y_pred) == len(y_true), 'Data length error.'
    all_acc = {}
    all_acc['total'] = np.around((y_pred%2 == y_true%2).sum()*100 / len(y_true), decimals=2)
    
    task_acc = []
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        label = '{}-{}'.format(str(class_id).rjust(2, '0'), str(class_id+increment-1).rjust(2, '0'))
        all_acc[label] = np.around(((y_pred[idxes]%2) == (y_true[idxes]%2)).sum()*100 / len(idxes), decimals=2)
        task_acc.append(np.around(((y_pred[idxes]%2) == (y_true[idxes]%2)).sum()*100 / len(idxes), decimals=2))
    all_acc['task_wise'] = sum(task_acc)/len(task_acc)
    return all_acc



args = setup_parser().parse_args()
param = load_json(args.config)
args = vars(args)  # Converting argparse Namespace to a dict.
args.update(param)  # Add parameters from json

if args["net_type"] == "slip":
    model = SliNet(args)
elif args["net_type"] == "sip":
    model = SiNet(args)
else:
    raise ValueError('Unknown net: {}.'.format(args["net_type"]))

checkpoint = torch.load(args["resume"])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"--- Task {checkpoint['tasks']} model loaded ---")


device = "cuda:0"
if not torch.cuda.is_available():
    device = "cpu"
model = model.to(device)
test_dataset = DummyDataset(args["dataroot"], args["datatype"], args["scenario"], args["compression"])
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=int(os.environ['SLURM_CPUS_ON_NODE']))

X,Y = [], []

for id, task_centers in enumerate(checkpoint["all_keys"]):
    X.append(task_centers.detach().cpu().numpy())
    Y.append(np.array([id]*len(task_centers)))

X = np.concatenate(X,0)
Y = np.concatenate(Y,0)
neigh = KNeighborsClassifier(n_neighbors=1, metric='l1')
neigh.fit(X, Y)

selectionsss = []

y_pred, y_true = [], []
for _, (path, inputs, targets) in tqdm(enumerate(test_loader), total=len(test_loader)):
    inputs = inputs.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        feature = model.extract_vector(inputs)
        selection = neigh.predict(feature.detach().cpu().numpy())
        if args["random_select"]:
            selection = np.random.randint(0, Y.max(), selection.shape)
        if args["til"]:
            selection = (targets/345).cpu().long().numpy()
            # selection = (targets/50).cpu().long().numpy()
            # selection = (targets/2).cpu().long().numpy()

        selectionsss.extend(selection)

        selection = torch.tensor(selection) #! in the original code they multiply for 0

        outputs = model.interface(inputs, selection)
    predicts = torch.topk(outputs, k=2, dim=1, largest=True, sorted=True)[1]
    y_pred.append(predicts.cpu().numpy())
    y_true.append(targets.cpu().numpy())

y_pred = np.concatenate(y_pred)
y_true = np.concatenate(y_true)
print(sum(selectionsss==((y_true/345).astype(int)))/(len(y_true)))
if args["datatype"] == 'deepfake':
    print(accuracy_binary(y_pred.T[0], y_true))
elif args["datatype"] == 'domainnet':
    print(accuracy_domain(y_pred.T[0], y_true))
elif args["datatype"] == 'core50':
    print(accuracy_core50(y_pred.T[0], y_true))