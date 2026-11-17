import os.path 

# Paths for VISPR dataset.
vispr_path = '/mnt/private/xwy/Balancing_privacy_and_pose/dataset/vispr'

# Paths for UCF101 dataset.
ucf101_path = '/mnt/private/xwy/Balancing_privacy_and_pose/dataset/UCF-101'
ucf101_class_mapping = os.path.join(ucf101_path, 'ucfTrainTestlist', 'action_classes.json')
vp_ucf101_private_label = os.path.join(ucf101_path, 'ucfTrainTestlist', 'VPUCF_annotations', 'vp_ucf101_privacy_attribute_label.csv')

# Paths for HMDB51 dataset.
hmdb51_frames_path = '/home/DATASET/HMDB51_frames'
pahmdb_path = '/home/DATASET/PA-HMDB51-master/PrivacyAttributes'
hmdb51_path = '/home/DATASET/hmdb51'
pahmdb_test_path = '/home/DATASET/PA-HMDB51-master/Privacy_test_data/'
hmdb51_class_mapping = os.path.join(hmdb51_path, 'hmdbTrainTestlist', 'action_51_classes.json')
vp_hmdb51_private_label = os.path.join(hmdb51_path, 'hmdbTrainTestlist', 'VPHMDB_annotations', 'vp_hmdb51_privacy_attribute_label.csv')

# General paths.
saved_models_dir = os.path.join('saved_models')
logs = os.path.join('logs')
training_logs = '/mnt/private/xwy/Balancing_privacy_and_pose/fa_pretraining/exp'

