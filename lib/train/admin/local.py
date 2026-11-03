class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/media/root123/date-5T/YK-Track_code/MLSTTrack-main'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/media/root123/date-5T/YK-Track_code/MLSTTrack-main/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/media/root123/date-5T/YK-Track_code/MLSTTrack-main/pretrained_networks'
        self.got10k_val_dir = '/media/root123/date-5T/FMTrack_dataset/got10k/val'
        self.lasot_lmdb_dir = '/media/root123/date-5T/FMTrack_dataset/lasot_lmdb'
        self.got10k_lmdb_dir = '/media/root123/date-5T/FMTrack_dataset/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/media/root123/date-5T/FMTrack_dataset/trackingnet_lmdb'
        self.coco_lmdb_dir = '/media/root123/date-5T/FMTrack_dataset/coco_lmdb'
        self.coco_dir = '/media/root123/date-5T/FMTrack_dataset/coco'
        self.lasot_dir = '/media/root123/date-5T/FMTrack_dataset/lasot'
        self.got10k_dir = '/media/root123/date-5T/FMTrack_dataset/got10k/train'
        self.trackingnet_dir = '/media/root123/date-5T/FMTrack_dataset/trackingnet'
        self.depthtrack_dir = '/media/root123/date-5T/FMTrack_dataset/depthtrack/train'
        # self.lasher_dir = '/media/root123/date-5T/FMTrack_dataset/lasher/train'
        self.visevent_dir = '/media/root123/date-5T/FMTrack_dataset/visevent/train'

        # # LasHeR and VTUAV
        # self.lasher_train_dir = '/media/root123/date-5T/FMTrack_dataset/LasHeR/trainingset'  # xyl 修改Lasher数据集的路径
        # self.lasher_test_dir = '/media/root123/date-5T/FMTrack_dataset/LasHeR/testingset'  # xyl 修改Lasher数据集的路径
        # self.vtuav_train_dir = '/media/root123/date-5T/FMTrack_dataset/VTUAV/trainingset'  # xyl 修改VTUAV数据集的路径
        # self.vtuav_test_dir = '/media/root123/date-5T/FMTrack_dataset/VTUAV/testingset'  # xyl 修改VTUAV数据集的路径
        # # 新增数据集必须添加
        # self.hial_train_dir = '/media/root123/date-5T/FMTrack_dataset/HiAL/trainingset'  # xyl 修改HiAl数据集的路径
        # self.hial_test_dir = '/media/root123/date-5T/FMTrack_dataset/HiAL/testingset'  # xyl 修改HiAl数据集的路径

        self.lasher_dir = '/media/root123/date-5T/FMTrack_dataset/LasHeR/trainingset'  # yk 修改Lasher数据集的路径
        self.lasher_test_dir = '/media/root123/date-5T/FMTrack_dataset/LasHeR/testingset'  # xyl 修改Lasher数据集的路径