import numpy as np

from colorama import Style, Fore

from ..utils import overlap_ratio, success_overlap, success_error

import pprint

class OPEBenchmark:
    """
    Args:
        result_path: result path of your tracker
                should the same format like VOT
    """
    def __init__(self, dataset):
        # dataset: type: UAVDataset {'video':UAVVideo,..}
        self.dataset = dataset

    def convert_bb_to_center(self, bboxes):

        return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                         (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T

    def convert_bb_to_norm_center(self, bboxes, gt_wh):
        return self.convert_bb_to_center(bboxes) / (gt_wh+1e-16)

    def eval_success(self, eval_trackers=None):
        """
        Args: 
            eval_trackers: list of tracker name or single tracker name
        Return:
            res: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        # print("eval_trackers")
        # print(eval_trackers)
        success_ret = {}
        for tracker_name in eval_trackers:
            success_ret_ = {}
            # print("tracker_name")
            # print(tracker_name)
            for video in self.dataset: #TODO
                # print("video.gt_traj")
                # print(video.gt_traj)
                gt_traj = np.array(video.gt_traj)
                # print("gt_traj")
                # print(gt_traj)
                for gt in gt_traj:
                    if(np.any(gt==np.nan)):
                        print("gt")
                        print(gt)
                if tracker_name not in video.pred_trajs:
                    # print("\n tracker_name not in video pred_trajs") # TODO
                    tracker_traj = video.load_tracker(self.dataset.tracker_path, #../cftracker# TODO
                            tracker_name, False)#
                    tracker_traj = np.array(tracker_traj)
                else:
                    print("\n tracker_name not in video.pred_trajs")
                    tracker_traj = np.array(video.pred_trajs[tracker_name])

                n_frame = len(gt_traj)
                # if hasattr(video, 'absent'):
                #     print("\n video has attribute absent")
                #     gt_traj = gt_traj[video.absent == 1]
                #     tracker_traj = tracker_traj[video.absent == 1]

                # for tt in tracker_traj:
                #     if(np.any(tt==[2])):
                #         print("tt")
                #         print(tt)

                modi_gt_traj = []
                modi_tracker_traj= []
                # target_lost=0
                skip_count = 0
                tracked_loss = 0
                for i in range(len(gt_traj)):
                    if(tracker_traj[i]==[1] or tracker_traj[i]==[0]):
                        skip_count =skip_count+1
                        continue#init,skip不用算success
                    elif (tracker_traj[i]==[3]):
                        tracked_loss = tracked_loss +1
                    elif tracker_traj[i]==[2]:
                        continue
                    else:
                        modi_gt_traj.append(gt_traj[i])
                        modi_tracker_traj.append(tracker_traj[i])

                modi_gt_traj = np.array(modi_gt_traj)
                modi_tracker_traj = np.array(modi_tracker_traj)

                if(modi_tracker_traj.shape!=modi_gt_traj.shape): # TODO
                    # print("modi_tracker_traj.shape,modi_gt_traj.shape")
                    # print(modi_tracker_traj.shape,modi_gt_traj.shape) #(32, 8) (32, 4)
                    # print("tracker_traj.shape")
                    # print(tracker_traj.shape)#(553,)
                    # print("tracker_traj")
                    # print(tracker_traj)
                    print("tracker-name,video-name")
                    print(tracker_name,video)
                    print("modi_tracker_traj")
                    #TODO:LDES追踪器一行有八个元素
                    print(modi_tracker_traj)#TODO:modi_tracker_traj 一行有8个元素，不知道为什么4
                    # print("modi_gt_traj")
                    # print(modi_gt_traj)
                    raise Exception("modi_tracker_traj.shape!=modi_gt_traj.shape")
                # print("modi_tracker_traj.shape,modi_gt_traj.shape")
                # print(modi_tracker_traj.shape,modi_gt_traj.shape)
                # 一个location都没有
                if(modi_tracker_traj==[]):
                    # print(target_lost,skip_count)
                    success_ret_[video.name] = [tracked_loss/n_frame]*len(np.arange(0, 1.05, 0.05))
                else:
                    success_ret_[video.name] = success_overlap(modi_gt_traj, modi_tracker_traj, n_frame,tracked_loss)
            success_ret[tracker_name] = success_ret_
        return success_ret

    def eval_precision(self, eval_trackers=None):
        """
        Args:
            eval_trackers: list of tracker name or single tracker name
        Return:
            res: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        precision_ret = {}
        for tracker_name in eval_trackers:
            precision_ret_ = {}
            for video in self.dataset:
                gt_traj = np.array(video.gt_traj)
                if tracker_name not in video.pred_trajs:
                    tracker_traj = video.load_tracker(self.dataset.tracker_path,
                            tracker_name, False)
                    tracker_traj = np.array(tracker_traj)
                else:
                    tracker_traj = np.array(video.pred_trajs[tracker_name])
                n_frame = len(gt_traj)

                modi_gt_traj = []
                modi_tracker_traj= []
                tracked_lost=0
                for i in range(len(gt_traj)):
                    if(tracker_traj[i]==[1] or tracker_traj[i]==[0]):
                        continue#init,skip不用算success
                    elif (tracker_traj[i]==[3]):
                        tracked_lost = tracked_lost+1
                    elif tracker_traj[i]==[2]:
                        continue
                    else:
                        modi_gt_traj.append(gt_traj[i])
                        modi_tracker_traj.append(tracker_traj[i])

                modi_gt_traj = np.array(modi_gt_traj)
                modi_tracker_traj = np.array(modi_tracker_traj)


                # print("modi_tracker_traj.shape,modi_gt_traj.shape")
                # print(modi_tracker_traj.shape,modi_gt_traj.shape) # TODO\
                if(modi_gt_traj.ndim==1):
                    # print(target_lost)
                    precision_ret_[video.name] = tracked_lost/n_frame
                else:
                    if(modi_gt_traj.ndim==1):
                        print("modi_gt_traj,modi_tracker_traj")
                        print(modi_gt_traj,modi_tracker_traj)
                        raise Exception("modi_gt_traj.ndim==1")
                    gt_center = self.convert_bb_to_center(modi_gt_traj)
                    tracker_center = self.convert_bb_to_center(modi_tracker_traj)

                    thresholds = np.arange(0, 51, 1)
                    precision_ret_[video.name] = success_error(gt_center, tracker_center,
                            thresholds, n_frame,tracked_lost)
            precision_ret[tracker_name] = precision_ret_
        return precision_ret

    def eval_norm_precision(self, eval_trackers=None):
        """
        Args:
            eval_trackers: list of tracker name or single tracker name
        Return:
            res: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        norm_precision_ret = {}
        for tracker_name in eval_trackers:
            norm_precision_ret_ = {}
            for video in self.dataset:
                gt_traj = np.array(video.gt_traj)
                if tracker_name not in video.pred_trajs:
                    tracker_traj = video.load_tracker(self.dataset.tracker_path, 
                            tracker_name, False)
                    tracker_traj = np.array(tracker_traj)
                else:
                    tracker_traj = np.array(video.pred_trajs[tracker_name])
                n_frame = len(gt_traj)
                if hasattr(video, 'absent'):
                    gt_traj = gt_traj[video.absent == 1]
                    tracker_traj = tracker_traj[video.absent == 1]
                gt_center_norm = self.convert_bb_to_norm_center(gt_traj, gt_traj[:, 2:4])
                tracker_center_norm = self.convert_bb_to_norm_center(tracker_traj, gt_traj[:, 2:4])
                thresholds = np.arange(0, 51, 1) / 100
                norm_precision_ret_[video.name] = success_error(gt_center_norm,
                        tracker_center_norm, thresholds, n_frame)
            norm_precision_ret[tracker_name] = norm_precision_ret_
        return norm_precision_ret

    def show_result(self, success_ret, precision_ret=None,
            norm_precision_ret=None, show_video_level=False, helight_threshold=0.6):
        """pretty print result
        Args:
            result: returned dict from function eval
        """
        # sort tracker
        tracker_auc = {}
        for tracker_name in success_ret.keys():
            auc = np.mean(list(success_ret[tracker_name].values()))
            tracker_auc[tracker_name] = auc
        tracker_auc_ = sorted(tracker_auc.items(),
                             key=lambda x:x[1],
                             reverse=True)[:20]
        tracker_names = [x[0] for x in tracker_auc_]


        tracker_name_len = max((max([len(x) for x in success_ret.keys()])+2), 12)
        header = ("|{:^"+str(tracker_name_len)+"}|{:^9}|{:^16}|{:^11}|").format(
                "Tracker name", "Success", "Norm Precision", "Precision")
        formatter = "|{:^"+str(tracker_name_len)+"}|{:^9.3f}|{:^16.3f}|{:^11.3f}|"

        print('-'*len(header))
        print(header)
        print('-'*len(header))

        for tracker_name in tracker_names:
            # success = np.mean(list(success_ret[tracker_name].values()))
            success = tracker_auc[tracker_name]
            if precision_ret is not None:
                precision = np.mean(list(precision_ret[tracker_name].values()), axis=0)[20]
            else:
                precision = 0
            if norm_precision_ret is not None:
                norm_precision = np.mean(list(norm_precision_ret[tracker_name].values()),
                        axis=0)[20]
            else:
                norm_precision = 0
            print(formatter.format(tracker_name, success, norm_precision, precision))
        print('-'*len(header))

        if show_video_level and len(success_ret) < 10 \
                and precision_ret is not None \
                and len(precision_ret) < 10:
            print("\n\n")
            header1 = "|{:^21}|".format("Tracker name")
            header2 = "|{:^21}|".format("Video name")
            for tracker_name in success_ret.keys():
                # col_len = max(20, len(tracker_name))
                header1 += ("{:^21}|").format(tracker_name)
                header2 += "{:^9}|{:^11}|".format("success", "precision")
            print('-'*len(header1))
            print(header1)
            print('-'*len(header1))
            print(header2)
            print('-'*len(header1))
            videos = list(success_ret[tracker_name].keys())

            for video in videos:
                row = "|{:^21}|".format(video)
                for tracker_name in success_ret.keys():
                    success = np.mean(success_ret[tracker_name][video])
                    precision = np.mean(precision_ret[tracker_name][video])
                    success_str = "{:^9.3f}".format(success)

                    if success < helight_threshold:
                        row += '{}{}{}|'.format(Fore.RED,success_str,Style.RESET_ALL)
                    else:
                        row += success_str+'|'
                    precision_str = "{:^11.3f}".format(precision)

                    if precision < helight_threshold:
                        row += '{}{}{}|'.format(Fore.RED,precision_str,Style.RESET_ALL)
                    else:
                        row += precision_str+'|'

                print(row)

            print('-'*len(header1))

    def show_result_new(self, success_ret, precision_ret):
        mean_success_ret = {}
        for tracker_key,tracker_value in success_ret.items():
            for video_key,video_value in tracker_value.items():
                # print("video_value.ndim")
                # print(video_value.ndim)
                if(video_value.ndim==0):
                    print(video_key,video_value.ndim)
                    raise Exception('video_value.ndim == 0')
                if video_key == 'bike2':
                    overall_success = video_value
                else:
                    overall_success = np.concatenate((overall_success,video_value),axis=0)

            mean_success_array = np.mean(overall_success,axis=0)
            mean_success_ret[tracker_key] = mean_success_array


        mean_precision_ret = {}
        for tracker_key, tracker_value in precision_ret.items():
            for video_key, video_value in tracker_value.items():
                # print(video_key,video_value)
                if isinstance(video_value,float):
                    continue
                if video_key == 'bike2':
                    overall_precision = video_value
                else:
                    overall_precision = np.concatenate((overall_precision, video_value), axis=0)

            mean_precision_array = np.mean(overall_precision, axis=0)
            mean_precision_ret[tracker_key] = mean_precision_array


