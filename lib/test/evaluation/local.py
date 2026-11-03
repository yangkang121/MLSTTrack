from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/media/root123/date-5T/FMTrack_dataset/got10k_lmdb'
    settings.got10k_path = '/media/root123/date-5T/FMTrack_dataset/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/media/root123/date-5T/FMTrack_dataset/itb'
    settings.lasot_extension_subset_path_path = '/media/root123/date-5T/FMTrack_dataset/lasot_extension_subset'
    settings.lasot_lmdb_path = '/media/root123/date-5T/FMTrack_dataset/lasot_lmdb'
    settings.lasot_path = '/media/root123/date-5T/FMTrack_dataset/lasot'
    settings.network_path = '/media/root123/date-5T/YK-Track_code/RSTTrack-main-uncertainty-memory-full/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/media/root123/date-5T/FMTrack_dataset/nfs'
    settings.otb_path = '/media/root123/date-5T/FMTrack_dataset/otb'
    settings.prj_dir = '/media/root123/date-5T/YK-Track_code/RSTTrack-main-uncertainty-memory-full'

    settings.result_plot_path = '/media/root123/date-5T/YK-Track_code/RSTTrack-main-uncertainty-memory-full/output/test/result_plots'
    settings.results_path = '/media/root123/date-5T/YK-Track_code/RSTTrack-main-uncertainty-memory-full/output/test/tracking_results'    # Where to store tracking results
    # /media/root123/date-5T/YK-Track_code/RSTTrack-main/lib/test/parameter/rsttrack.py中8-24行来判断加载哪个数据集的权重
    settings.save_dir = '/media/root123/date-5T/YK-Track_code/RSTTrack-main-uncertainty-memory-full/output'


    settings.segmentation_path = '/media/root123/date-5T/YK-Track_code/RSTTrack-main-uncertainty-memory-full/output/test/segmentation_results'
    settings.tc128_path = '/media/root123/date-5T/FMTrack_dataset/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/media/root123/date-5T/FMTrack_dataset/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/media/root123/date-5T/FMTrack_dataset/trackingnet'
    settings.uav_path = '/media/root123/date-5T/FMTrack_dataset/uav'
    settings.vot18_path = '/media/root123/date-5T/FMTrack_dataset/vot2018'
    settings.vot22_path = '/media/root123/date-5T/FMTrack_dataset/vot2022'
    settings.vot_path = '/media/root123/date-5T/FMTrack_dataset/VOT2019'
    settings.youtubevos_dir = ''

    # yk LasHeR/VTUAV/HiAL测试路径
    settings.lasher_path = '/media/root123/date-5T/FMTrack_dataset/LasHeR'
    settings.lashertestingset_path = '/media/root123/date-5T/FMTrack_dataset/LasHeR/testingset'
    settings.vtuav_path = '/media/root123/date-5T/FMTrack_dataset/VTUAV'
    settings.vtuavtestingset_dir = '/media/root123/date-5T/FMTrack_dataset/VTUAV/testingset'
    settings.hial_path = '/media/root123/date-5T/FMTrack_dataset/HiAL'
    settings.hial_test_dir = '/media/root123/date-5T/FMTrack_dataset/HiAL/testingset'
    # yk RGBT210/234测试
    settings.rgbt210_dir = '/media/root123/date-5T/FMTrack_dataset/RGBT-210/RGBT_T210'
    settings.rgbt234_dir = '/media/root123/date-5T/FMTrack_dataset/RBGT-234/RGBT_T234'
    settings.gtot_dir = '/media/root123/date-5T/FMTrack_dataset/GTOT'


    return settings

