from tqdm import tqdm

class Dataset(object):
    def __init__(self, name, dataset_root):
        '''

        :param name:
        :param dataset_root: dataset_root:  pyCFTrackers/dataset/UAV123
        '''
        self.name = name
        self.dataset_root = dataset_root
        self.videos = None

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.videos[idx]
        elif isinstance(idx, int):
            return self.videos[sorted(list(self.videos.keys()))[idx]]

    def __len__(self):
        return len(self.videos)

    def __iter__(self):
        keys = sorted(list(self.videos.keys()))
        for key in keys:
            yield self.videos[key]

    def set_tracker(self, path, tracker_names):
        """
        Args:
            path: path to tracker results, ../cftrackers
            tracker_names: list of tracker name <- tracker name
            caller: eval_UAV123
        """
        self.tracker_path = path
        self.tracker_names = tracker_names
        # for video in tqdm(self.videos.values(), 
        #         desc='loading tacker result', ncols=100):
        #     video.load_tracker(path, tracker_names)
