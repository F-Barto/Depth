from networks.legacy.packnet.packnet import PackNet01
from networks.legacy.packnet.posenet import PoseNet
from networks.legacy.monodepth2.depth_rest_net import DepthResNet
from networks.legacy.monodepth2.pose_res_net import PoseResNet as PoseResNet_legacy
from networks.legacy.monodepth2.guided_depth_rest_net import GuidedDepthResNet
from networks.legacy.custom.guided_sparse_dilated_depth_net import GuidedSparseDepthResNet
from networks.legacy.monodepth_original.depth_res_net import DepthResNet as OriginalDepthResNet
from networks.legacy.monodepth_original.pose_res_net import PoseResNet as OriginalPoseResNet

from networks.nets.depth_nets.monodepth2 import DepthNetMonodepth2
from networks.nets.pose_nets.monodepth2 import PoseResNet

def select_depth_net(depth_net_name, depth_net_options, input_channels=3, load_sparse_depth=False):

    sparse_depth_input_required = ['guiding', 'sparse-guiding']
    if depth_net_name in sparse_depth_input_required:
        assert load_sparse_depth, "Sparse depth signal is necessary for feature guidance."

    if depth_net_name == 'packnet':
        depth_net = PackNet01
    elif depth_net_name == 'monodepth':
        depth_net = DepthResNet
    elif depth_net_name == 'monodepth2':
        depth_net = DepthNetMonodepth2
    elif depth_net_name == 'monodepth_original':
        depth_net = OriginalDepthResNet
    elif depth_net_name == 'guiding':
        depth_net = GuidedDepthResNet
    elif depth_net_name == 'sparse-guiding':
        depth_net = GuidedSparseDepthResNet
    else:
        raise NotImplementedError(f"Depth network of name {depth_net_name} not implemented")

    if depth_net_name != 'monodepth_original':
        depth_net_options['input_channels'] = input_channels

    return depth_net(**depth_net_options)

def select_pose_net(pose_net_name, pose_net_options):
    if pose_net_name == 'packnet':
        pose_net = PoseNet
    elif pose_net_name == 'monodepth':
        pose_net = PoseResNet_legacy
    elif pose_net_name == 'monodepth2':
        pose_net = PoseResNet
    elif pose_net_name == 'monodepth_original':
        pose_net = OriginalPoseResNet
    else:
        raise NotImplementedError(f"Pose network of name  {pose_net_name} not implemented")

    return pose_net(**pose_net_options)


