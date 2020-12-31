from __future__ import division
import argparse
import logging
import numpy as np
import cv2
from os import makedirs
from os.path import join, isdir

from lib.log_helper import init_log, add_file_handler
from lib.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect
from lib.benchmark_helper import load_dataset


from lib.pysot.utils import region
from cftracker.mosse import MOSSE
from cftracker.staple import Staple
from cftracker.dsst import DSST
from cftracker.samf import SAMF
from cftracker.kcf import KCF
from cftracker.csk import CSK
from cftracker.cn import CN
from cftracker.dat import DAT
from cftracker.eco import ECO
from cftracker.bacf import BACF
from cftracker.csrdcf import CSRDCF
from cftracker.ldes import LDES
from cftracker.mkcfup import MKCFup
from cftracker.strcf import STRCF
from cftracker.mccth_staple import MCCTHStaple
from cftracker.opencv_cftracker import OpenCVCFTracker
from cftracker.ReIDkcf import ReIDKCF

# from lib.eco.config import vot16_deep_config,vot16_hc_config
from cftracker.config import ldes_config,dsst_config,csrdcf_config,staple_config,mkcf_up_config,mccth_staple_config

import time

'''
get the comparision result with ground truth and prediction location 

'''





parser = argparse.ArgumentParser(description='Test')

parser.add_argument('--dataset', dest='dataset', default='UAV123',
                    help='datasets')
parser.add_argument('-l', '--log', default="log_test_uav123.txt", type=str, help='log file')
parser.add_argument('-v', '--visualization', dest='visualization', action='store_true',
                    help='whether visualize result',default=True)
parser.add_argument('--gt', action='store_true', help='whether use gt rect for davis (Oracle)')

def create_tracker(tracker_type):
    if tracker_type == 'MOSSE':
        tracker = MOSSE()
    elif tracker_type == 'CSK':
        tracker = CSK()
    elif tracker_type == 'CN':
        tracker = CN()
    elif tracker_type == 'DSST':
        tracker = DSST(dsst_config.DSSTConfig())
    elif tracker_type=='DSST-LP':
        tracker=DSST(dsst_config.DSSTLPConfig())
    elif tracker_type=='SAMF':
        tracker=SAMF()
    elif tracker_type == 'Staple':
        tracker = Staple(config=staple_config.StapleVOTConfig())
    #elif tracker_type=='Staple-CA':
    #    tracker=Staple(config=staple_config.StapleCAVOTConfig())
    elif tracker_type == 'KCF':
        tracker = KCF(features='hog', kernel='gaussian')
    elif tracker_type == 'DCF':
        tracker = KCF(features='hog', kernel='linear')
    elif tracker_type == 'DAT':
        tracker = DAT()
    # elif tracker_type=='ECO-HC':
    #     tracker=ECO(config=vot16_hc_config.VOT16HCConfig())
    # elif tracker_type=='ECO':
    #     tracker=ECO(config=vot16_deep_config.VOT16DeepConfig())
    elif tracker_type=='BACF':
        tracker=BACF()
    elif tracker_type=='CSRDCF':
        tracker=CSRDCF(csrdcf_config.CSRDCFConfig())
    elif tracker_type=='CSRDCF-LP':
        tracker=CSRDCF(csrdcf_config.CSRDCFLPConfig())
    elif tracker_type=='OPENCV_KCF':
        tracker=OpenCVCFTracker(name='KCF')
    elif tracker_type=='OPENCV_MOSSE':
        tracker=OpenCVCFTracker(name='MOSSE')
    elif tracker_type=='OPENCV-CSRDCF':
        tracker=OpenCVCFTracker(name='CSRDCF')
    elif tracker_type=='LDES':
        tracker=LDES(config=ldes_config.LDESVOTLinearConfig())
    elif tracker_type=='LDES-NoBGD':
        tracker=LDES(config=ldes_config.LDESVOTNoBGDLinearConfig())
    elif tracker_type=='MKCFup':
        tracker=MKCFup(config=mkcf_up_config.MKCFupConfig())
    elif tracker_type=='MKCFup-LP':
        tracker=MKCFup(config=mkcf_up_config.MKCFupLPConfig())
    elif tracker_type=='STRCF':
        tracker=STRCF()
    elif tracker_type=='MCCTH-Staple':
        tracker=MCCTHStaple(config=mccth_staple_config.MCCTHVOTConfig())
    elif tracker_type == "ReIDKCF":
        tracker = ReIDKCF(features='hog', kernel='gaussian')#TODO
    else:
        raise NotImplementedError
    return tracker

def track_uav(tracker_type, video): #TODO 用tracker进行追踪得到追踪结果
    '''

    :param tracker_type:
    :param video:
    :return:
    在指定的video序列上，运行tracker_type的tracker
    将regions结果,一行一行写进得到 ./UAV123/tracker-type/baseline/bird1/video_name_001.txt文件中
    '''

    regions = []  # result and states[1 init / 2 lost / 0 skip]
    image_files, gt = video['image_files'], video['gt']

    start_frame, end_frame, lost_times, toc = 0, len(image_files), 0, 0

    #f frame index
    for f, image_file in enumerate(image_files):

        #print(image_file)
        im = cv2.imread(image_file)
        #print(im)
        tic = cv2.getTickCount()
        if np.any(im==None):
            print (f,image_file)
            raise Exception("fail to read image ")

        if f == 0 :  # init
            # print(f)
            # print("f==start_frame branch")

            tracker=create_tracker(tracker_type)
            if tracker_type=='LDES':
                tracker.init(im,gt[f])
                if tracker.polygon is True:
                    location=gt[f]
                else:
                    cx, cy, w, h = get_axis_aligned_bbox(gt[f])
                    target_pos = np.array([cx, cy])
                    target_sz = np.array([w, h])
                    location = cxy_wh_2_rect(target_pos, target_sz)

            else:
                #center x center y
                cx, cy, w, h = get_axis_aligned_bbox(gt[f])
                target_pos = np.array([cx, cy])
                target_sz = np.array([w, h])
                location=cxy_wh_2_rect(target_pos,target_sz)
                # print("im.shape")
                # print(im.shape)
                tracker.init(im,((cx-w/2),(cy-h/2),(w),(h)))
            #1 是初始化
            regions.append(1 if 'UAV' in args.dataset else gt[f]) # TODO

        elif f > start_frame:
            # print(f)
            # print("update the location ")
            if tracker_type == "ReIDKCF":
                location=tracker.update(im,f,image_file)
            else:
                location = tracker.update(im, f)

            if tracker_type=="ReIDKCF" and location==None:
                if np.any(gt[f])==np.nan:
                    regions.append(3)#tracked loss
                else:
                    regions.append(2)#2追踪失败
                    lost_times += 1
                    start_frame = f + 5  # skip 2 frames
            else:
                if 'UAV' in args.dataset:
                    b_overlap = region.vot_overlap(gt[f],location, (im.shape[1], im.shape[0]))
                else:
                    b_overlap = 1

                if b_overlap:
                    #有重合部分 location
                    regions.append(location)
                else:  # 2 追踪失败，lost fail to track
                    regions.append(2)
                    lost_times += 1
                    start_frame = f + 5  # skip 2 frames

        else:  # skip
            # 0 跳过去
            regions.append(0)

        toc += cv2.getTickCount() - tic

        if args.visualization and f > start_frame:  # visualization (skip lost frame)
            im_show = im.copy()
            if f == 0: cv2.destroyAllWindows()

            if gt.shape[0] > f and np.all(gt[f] != np.nan):

                if len(gt[f]) == 8:
                    cv2.polylines(im_show, [np.array(gt[f], np.int).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
                else:
                    # print("(gt[f, 0], gt[f, 1]), (gt[f, 0] + gt[f, 2], gt[f, 1] + gt[f, 3])")
                    # print((gt[f, 0], gt[f, 1]), (gt[f, 0] + gt[f, 2], gt[f, 1] + gt[f, 3]))
                    # print("im_show")
                    # print(im_show.shape)

                    cv2.rectangle(im_show, (int(gt[f, 0]), int(gt[f, 1])), (int(gt[f, 0] + gt[f, 2]), int(gt[f, 1] + gt[f, 3])), (0, 255, 0), 3)

            # draw prediction rectangle
            if len(location) == 8:
                location_int = np.int0(location)
                cv2.polylines(im_show, [location_int.reshape((-1, 1, 2))], True, (0, 255, 255), 3)
            else:
                location = [int(l) for l in location]
                cv2.rectangle(im_show, (int(location[0]), int(location[1])),
                              (int(location[0] + location[2]), int(location[1] + location[3])), (0, 255, 255), 3)

            cv2.putText(im_show, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(im_show, str(lost_times), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            #cv2.imshow(video['name'], im_show)
            cv2.waitKey(1)



    toc /= cv2.getTickFrequency()

    # save result
    name = tracker_type

    #UAV123/tracker-type/baseline/bird1
    video_path = join(args.dataset, name,
                          'baseline', video['name'])
    # print(video_path)
    if not isdir(video_path): makedirs(video_path)
    result_path = join(video_path, '{:s}_001.txt'.format(video['name']))
    # print("result_path get_uav_123 line174")
    # print(result_path)
    with open(result_path, "w") as fin:
        for x in regions:
            fin.write("{:d}\n".format(x)) if isinstance(x, int) else \
            fin.write(','.join([region.vot_float2str("%.4f", i) for i in x]) + '\n')

    logger.info('({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps Lost: {:d} Tracker:{}'.format(
        v_id, video['name'], toc, f / toc, lost_times,tracker_type))

    return lost_times, f / toc


def main():
    '''

    :return:
    '''
    global args, logger, v_id
    args = parser.parse_args()
    init_log('global', logging.INFO)
    if args.log != "":
        add_file_handler('global', args.log, logging.INFO)

    logger = logging.getLogger('global')
    logger.info(args)

    print("start load dataset ")
    # setup dataset
    #UAV123 info{video:{image_file:[],gt:[],name:'' }, }
    dataset = load_dataset(args.dataset)


    total_lost = 0
    speed_list = []

    # trackers = ['KCF', 'DCF','DAT','SAMF','MOSSE','CSK','CN','BACF','CSRDCF','LDES','MKCFup','Staple']
    trackers = ['KCF', 'DCF','ReIDKCF']
    # trackers = ['ReIDKCF']
    # trackers = []
    # trackers = ['SAMF','CN','DAT','BACF']
    #switch
    flag1 = False
    flag2 = False
    # starting_video = "group2_2 "
    # start_tracker = "DAT"

    for tracker_type in trackers:
        # if tracker_type == start_tracker:
        #     flag2 = True
        #
        # if flag2 ==False:
        #     continue

        total_lost = 0
        speed_list = []

        start_time = time.time()

        print("tracker_type",tracker_type)
        for v_id, video in enumerate(dataset.keys(), start=1):
            # if video == starting_video:
            #     flag1 = True
            # if flag1 == False:
            #     continue
            # if(video not in ["group1_2","car10","person8_1","person12_2","car3_s","car1_1","group3_1","car11","person13","group1_3","person6","group1_1","person11"]):
            #     continue
            print("video",video)
            # if(video!="person14_2"):
            #     continue
            lost, speed = track_uav(tracker_type,dataset[video])
            total_lost += lost
            speed_list.append(speed)

        end_time = time.time()
        logger.info('elapsed time :{}'.format(end_time-start_time))
        logger.info('Total Lost: {:d}'.format(total_lost))

        logger.info('Mean Speed: {:.2f} FPS'.format(np.mean(speed_list)))


if __name__ == '__main__':
    main()
