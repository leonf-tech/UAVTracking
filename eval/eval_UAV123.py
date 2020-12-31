import argparse
import glob
from os.path import join, realpath, dirname

from tqdm import tqdm
from multiprocessing import Pool
#from lib.pysot.datasets import OTBDataset
from lib.pysot.datasets import UAVDataset
from lib.pysot.evaluation import OPEBenchmark
from lib.pysot.visualization import draw_success_precision
import pprint
import pickle
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VOT Evaluation')
    parser.add_argument('--dataset', type=str, default='UAV123',help='dataset name')
    parser.add_argument('--tracker_dir', type=str, default='../cftracker',help='tracker result root')
    #parser.add_argument('--tracker_prefix', type=str,default='test', help='tracker prefix')
    parser.add_argument('--show_video_level', action='store_true')
    parser.add_argument('--num', type=int, help='number of processes to eval', default=10)
    parser.add_argument('--vis',type=bool,default=True)
    args = parser.parse_args()

    #dataset_root pyCFTrackers/dataset/UAV123
    root = '../dataset/UAV123'
    #root = join(realpath(dirname(__file__)), '/dataset/UAV123')
    #../cftracker
    tracker_dir = args.tracker_dir
    trackers = glob.glob(tracker_dir)
    trackers = [t.split('/')[-1] for t in trackers]
    trackers = ['KCF', 'DCF','DAT','SAMF','MOSSE','CSK','CN','BACF','CSRDCF','MKCFup','Staple','ReIDKCF'] # TODO
    # trackers = ['CSK'] # TODO
    print(trackers)
    assert len(trackers) > 0
    # args.num = min(args.num, len(trackers)) # TODO
    args.num = 1
    if 'UAV' in args.dataset:

        dataset = UAVDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)

        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success, #TODO
                                                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)


        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)

        pprint.pprint(success_ret)#TODO: 不同的tracker success, precision 结果一样

        with open("./results/success_ret_result",'wb') as f:
            pickle.dump(success_ret,f)

        with open("./results/precision_ret_result",'wb') as f:
            pickle.dump(precision_ret,f)
        #
        # with open("./results/success_ret_result",'rb') as f:
        #     success_ret = pickle.load(f)
        #
        # with open("./results/precision_ret_result",'rb') as f:
        #     precision_ret = pickle.load(f)

        # pprint.pprint(success_ret)
        # benchmark.show_result_new(success_ret, precision_ret)
        benchmark.show_result(success_ret, precision_ret)
        #
        #
        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                                       name=dataset.name,
                                       videos=videos,
                                       attr=attr,
                                       precision_ret=precision_ret)
