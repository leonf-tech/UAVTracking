import os
from examples.pytracker import PyTracker
from lib.utils import get_ground_truthes,plot_precision,plot_success
# from examples.otbdataset_config import OTBDatasetConfig
if __name__ == '__main__':
    # module_path = os.path.abspath(__file__)
    # dir_path = os.path.dirname(module_path.split("/"))
    data_dir='../dataset/UAV123'
    #name of the sequence
    data_names=sorted(os.listdir(data_dir))
    data_names = ["person8_1"]
    print(data_names)
    # dataset_config=OTBDatasetConfig()
    for data_name in data_names:
        # print(data_name)
        if data_name not in ["person8_1"]:
            continue
        print(data_name)
        #../dataset/test/person1
        data_path=os.path.join(data_dir,data_name)
        #[[<gt of a frame>],[],..]
        gts = get_ground_truthes(data_path)
        # if data_name in dataset_config.frames.keys():
        #     start_frame,end_frame=dataset_config.frames[data_name][:2]
        #
        #print(gts)
        #../dataset/test/person1/img
        img_dir = os.path.join(data_path,'img')
        tracker = PyTracker(img_dir,tracker_type='ReIDKCF')
        #print("okay tracker")
        poses=tracker.tracking(verbose=True,video_path=os.path.join('../results/CF/ReIDKCF',data_name+'_vis.avi'))
        #print("okay poses")
        # plot_success(gts,poses,os.path.join('../results/CF/ECO-HC',data_name+'_success.jpg'))
        # plot_precision(gts,poses,os.path.join('../results/CF/ECO-HC',data_name+'_precision.jpg'))
