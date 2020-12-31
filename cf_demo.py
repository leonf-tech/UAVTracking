import os
from examples.pytracker import PyTracker
from lib.utils import get_ground_truthes,plot_precision,plot_success
from examples.otbdataset_config import OTBDatasetConfig
if __name__ == '__main__':

    data_dir= './dataset/test'
    #print(os.listdir(data_dir))
    #name of the sequence
    data_names=sorted(os.listdir(data_dir))

    print(data_names)
    dataset_config=OTBDatasetConfig()
    for data_name in data_names:
        if data_name in ["person23"]:
            #../dataset/test/person1
            data_path=os.path.join(data_dir,data_name)
            #[[<gt of a frame>],[],..]
            gts = get_ground_truthes(data_path)

            if data_name in dataset_config.frames.keys():
                start_frame,end_frame=dataset_config.frames[data_name][:2]
                if data_name != 'David':
                    gts=gts[start_frame-1:end_frame]
            #print(gts)

            #../dataset/test/person1/img
            img_dir = os.path.join(data_path,'img')
            tracker_type = "MCCTH-Staple"
            tracker = PyTracker(img_dir,tracker_type=tracker_type,dataset_config=dataset_config)
            #print("okay tracker")
            print(tracker_type,"start tracking")
            poses=tracker.tracking(verbose=True,video_path=os.path.join("./results/CF/",tracker_type+"/",tracker_type+"_"+data_name+'_vis.avi'))
            print(tracker_type,"finish tracking")
            thresh1,success = plot_success(gts,poses,os.path.join("./results/CF/",tracker_type+"/",tracker_type+"_"+data_name+'_success.jpg'))
            thresh2,precision = plot_precision(gts,poses,os.path.join("./results/CF/",tracker_type+"/",tracker_type+"_"+data_name+'_precision.jpg'))

            with open(os.path.join("./results/CF/",tracker_type+"/","precision_result.txt"), "w") as fp:
                fp.write("threshold"+'\n')
                fp.write(str(thresh1)+'\n')
                fp.write("precision"+'\n')
                fp.write(str(precision)+'\n')

            with open(os.path.join("./results/CF/",tracker_type+"/","success_result.txt"), "w") as fp:
                fp.write("threshold"+'\n')
                fp.write(str(thresh2)+'\n')
                fp.write("success"+'\n')
                fp.write(str(success)+'\n')