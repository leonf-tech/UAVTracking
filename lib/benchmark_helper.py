# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from os.path import join, realpath, dirname, exists, isdir
from os import listdir
import logging
import glob
import numpy as np
import json
from collections import OrderedDict
import json
import re

def get_dataset_zoo():
    root = realpath(join(dirname(__file__), '../dataset'))
    zoos = listdir(root)

    def valid(x):
        y = join(root, x)
        if not isdir(y): return False

        return exists(join(y, 'list.txt')) \
               or exists(join(y, 'train', 'meta.json'))\
               or exists(join(y, 'ImageSets', '2016', 'val.txt'))

    zoos = list(filter(valid, zoos))
    return zoos


dataset_zoo = get_dataset_zoo()


def load_dataset(dataset):
    '''

    :param dataset: UAV123
    :return:
    '''
    info = OrderedDict()
    if 'UAV' in dataset:
        base_path = join('../dataset', dataset)
        if not exists(base_path):
            logging.error("Please download test dataset!!!")
            exit()
        # list_path = join(base_path, 'list.txt')
        # with open(list_path) as f:
        #     videos = [v.strip() for v in f.readlines()]
        #videos: list of the name of video sequence
        #../dataset/UAV123/UAV123.json
        with open(join(base_path,"UAV123.json")) as f:
            config_dict = json.load(f)
            # print(config_dict.keys())
            # print(json.dumps(config_dict,indent=4))
            videos = config_dict.keys()
            # print("print(len(config_dict['person8_1']['img_names']))")
            # print(len(config_dict['person8_1']['img_names']))
        #  针对group1_1等序列，配置gt部分，对应的groundtruth.txt文件
        # info[video] = {'image_files': image_files, 'gt': gt, 'name': video}

        for video in videos:
            # regex = '.+_\d'
            # compile_obj = re.compile(regex)
            # if compile_obj.findall(video) !=[]:
            #     video_path = join(base_path,video.split("_")[0])
            #     gt_path = join(video_path, video+'.txt')
            # else:
            #     video_path = join(base_path, video)
            #     gt_path = join(video_path, 'groundtruth_rect.txt')
            # [ base_path/video/img/000001.jpg，。。。
            image_names = config_dict[video]["img_names"]
            image_files = []
            for name in image_names:
                image_files.append(join(base_path,name.split("/")[0],"img",name.split("/")[-1]))
            # print(image_files)
            #[full-name1,...,]
            # image_files = sorted(glob.glob(image_path))
            gt = np.array(config_dict[video]["gt_rect"])#  convert to numpy array
            init_rect = config_dict[video]["init_rect"]
            attr = config_dict[video]["attr"]
            #print(gt)
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video,"init_rect":init_rect,"attr":attr}


    else:
        logging.error('Not support')
        exit()
    return info
