import os
import os.path
import sys
import logging
import copy
import time
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters


def train(args):
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])

    for seed in seed_list: # seed is for dataloader, applied only if shuffle param = True
        args['seed'] = seed
        args['device'] = device
        _train(args)


def _train(args):
    logfilename = 'logs/{}/{}_{}_{}_{}_{}_{}_{}_'.format(os.environ['SLURM_JOB_NAME'], args['prefix'], args['seed'], args['model_name'], args['net_type'],
                                                args['dataset'], args['init_cls'], args['increment'])+ time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    os.makedirs(logfilename)
    print(logfilename)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '/info.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("CUDA is available") if torch.cuda.is_available() else None

    _set_random(args)
    _set_device(args)
    print_args(args)
    data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'], args)
    args['class_order'] = data_manager._class_order
    args['filename'] = os.path.join(logfilename, "task")
    model = factory.get_model(args['model_name'], args)

    cnn_curve, nme_curve = {'top1': []}, {'top1': []}
    for task in range(data_manager.nb_tasks):
        logging.info('All params: {}'.format(count_parameters(model._network)))
        logging.info('Trainable params: {}'.format(count_parameters(model._network, True)))
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        if nme_accy is not None:
            logging.info('CNN: {}'.format(cnn_accy['grouped']))
            logging.info('NME: {}'.format(nme_accy['grouped']))
            cnn_curve['top1'].append(cnn_accy['top1'])
            nme_curve['top1'].append(nme_accy['top1'])
            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
            logging.info('NME top1 curve: {}'.format(nme_curve['top1']))
        else:
            logging.info('CNN: {}'.format(cnn_accy['grouped']))
            cnn_curve['top1'].append(cnn_accy['top1'])
            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))

        #torch.save(model, os.path.join(logfilename, "task_{}.pth".format(int(task))))
        model.save_checkpoint()

def _set_device(args):
    device_type = args['device']
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))

        gpus.append(device)

    args['device'] = gpus


def _set_random(args):
    torch.manual_seed(args['torch_seed'])
    torch.cuda.manual_seed(args['torch_seed'])
    torch.cuda.manual_seed_all(args['torch_seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))
