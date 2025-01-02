import sys
# sys.path.append('/data/Pcmaploc/code/geo-localization-with-point-clouds-and-openstreetmap')
from mmdet.registry import MODELS 
from mmengine.config import Config
# from mmdetection.mmdet.models.backbones import SwinTransformer

encoders = Config({
        'camera': {
            'backbone': {
                'type': 'SwinTransformer',
                'embed_dims': 96,
                'depths': [2, 2, 6, 2],
                'num_heads': [3, 6, 12, 24],
                'window_size': 7,
                'mlp_ratio': 4,
                'qkv_bias': True,
                'qk_scale': None,
                'drop_rate': 0.0,
                'attn_drop_rate': 0.0,
                'drop_path_rate': 0.3,
                'patch_norm': True,
                'out_indices': [1, 2, 3],
                'with_cp': False,
                'convert_weights': True,
                'init_cfg': {
                    'type': 'Pretrained',
                    'checkpoint': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
                }
            },
            'neck': {
                'type': 'GeneralizedLSSFPN',
                'in_channels': [192, 384, 768],
                'out_channels': 256,
                'start_level': 0,
                'num_outs': 3,
                'norm_cfg': {
                    'type': 'BN2d',
                    'requires_grad': True
                },
                'act_cfg': {
                    'type': 'ReLU',
                    'inplace': True
                },
                'upsample_cfg': {
                    'mode': 'bilinear',
                    'align_corners': False
                }
            },
            'vtransform': {
                'type': 'LSSTransform',
                'in_channels': 256,
                'out_channels': 80,
                'image_size': '${image_size}',
                'feature_size': '${[image_size[0] // 8, image_size[1] // 8]}',
                'xbound': [-51.2, 51.2, 0.4],
                'ybound': [-51.2, 51.2, 0.4],
                'zbound': [-10.0, 10.0, 20.0],
                'dbound': [1.0, 60.0, 0.5],
                'downsample': 2
            }
        },
        'lidar': {
            'voxelize': {
                'max_num_points': 10,
                'point_cloud_range': '${point_cloud_range}',
                'voxel_size': '${voxel_size}',
                'max_voxels': [90000, 120000]
            },
            'backbone': {
                'type': 'SparseEncoder',
                'in_channels': 5,
                'sparse_shape': [1024, 1024, 41],
                'output_channels': 128,
                'order': ['conv', 'norm', 'act'],
                'encoder_channels': [
                    [16, 16, 32],
                    [32, 32, 64],
                    [64, 64, 128],
                    [128, 128]
                ],
                'encoder_paddings': [
                    [0, 0, 1],
                    [0, 0, 1],
                    [0, 0, [1, 1, 0]],
                    [0, 0]
                ],
                'block_type': 'basicblock'
            }
    }
})

backbone=MODELS.build(encoders["camera"]["backbone"])
print(backbone)