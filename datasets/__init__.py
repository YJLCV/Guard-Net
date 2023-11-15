# from .kitti_dataset_1215 import KITTIDataset
from .kitti_dataset_1215_augmentation import KITTIDataset
# from .kitti_dataset_4mask_AA_crop import KITTIDataset
# from .sceneflow_dataset import SceneFlowDatset
from .sceneflow_dataset import SceneFlowDatset
#from .attention_conv import SceneFlowDatset
from .middlebury_data_our import MiddleburyStereoDataset

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    "middlebury":MiddleburyStereoDataset
}
