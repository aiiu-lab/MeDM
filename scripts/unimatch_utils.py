from unimatch.unimatch import UniMatch
from unimatch.geometry import forward_backward_consistency_check
import argparse
import torch
import numpy as np


def get_unimatch_model(weights_path):
    args = argparse.Namespace(
        reg_refine=True,
        num_reg_refine=6,
        weights_path=weights_path,
        feature_channels=128,
        num_scales=2,
        upsample_factor=4,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        task='flow',
        padding_factor=32,
        attn_type='swin',
        attn_splits_list=[2, 8],
        corr_radius_list=[-1, 4],
        prop_radius_list=[-1, 1],
        batch_size=2
    )
    unimatch = UniMatch(feature_channels=args.feature_channels,
                     num_scales=args.num_scales,
                     upsample_factor=args.upsample_factor,
                     num_head=args.num_head,
                     ffn_dim_expansion=args.ffn_dim_expansion,
                     num_transformer_layers=args.num_transformer_layers,
                     reg_refine=args.reg_refine,
                     task=args.task)
    ckpt = torch.load(args.weights_path, map_location='cpu')
    unimatch.load_state_dict(ckpt['model'])
    unimatch.eval()
    
    return unimatch, args

def get_flow(unimatch, rgbs, args, long_range_flow_offset=None):
    bs = args.batch_size
    device = next(unimatch.parameters()).device
    pred_flows = []
    for i in range(0, len(rgbs)-1, bs):
        image1 = rgbs[i:min(i+bs, len(rgbs)-1)].to(device)
        if long_range_flow_offset is None:
            image2 = rgbs[i+1:i+bs+1].to(device)
        else:
            anchors = torch.arange(i, i+len(image1)) + long_range_flow_offset
            anchors = anchors.clamp(max=len(rgbs)-1)
            image2 = rgbs[anchors].to(device)
            
        original_size = image1.shape[-2:]
        nearest_size = [int(np.ceil(image1.size(-2) / args.padding_factor)) * args.padding_factor,
                        int(np.ceil(image1.size(-1) / args.padding_factor)) * args.padding_factor]
        image1 = torch.nn.functional.interpolate(image1, size=nearest_size, mode='bilinear', align_corners=True)
        image2 = torch.nn.functional.interpolate(image2, size=nearest_size, mode='bilinear', align_corners=True)

        with torch.no_grad():
            results_dict = unimatch(image1, image2,
                                    attn_type=args.attn_type,
                                    attn_splits_list=args.attn_splits_list,
                                    corr_radius_list=args.corr_radius_list,
                                    prop_radius_list=args.prop_radius_list,
                                    num_reg_refine=args.num_reg_refine,
                                    task=args.task,
                                    pred_bidir_flow=True
                                    )
        pred_flow = results_dict['flow_preds'][-1]
        pred_flow = pred_flow.unflatten(0, (2, -1))
        pred_flows.append(pred_flow.cpu())
    pred_flows = torch.cat(pred_flows, dim=1)
    pred_flows = pred_flows.flatten(0, 1)
    pred_flows = torch.nn.functional.interpolate(pred_flows, size=original_size, mode='bilinear', align_corners=True)
    pred_flows[:, 0] = pred_flows[:, 0] * original_size[0] / nearest_size[0]
    pred_flows[:, 1] = pred_flows[:, 1] * original_size[1] / nearest_size[1]
    pred_flows = pred_flows.unflatten(0, (2, -1))
    
    pred_flows, pred_flows_back = pred_flows
    pred_occs, _ = forward_backward_consistency_check(pred_flows, pred_flows_back)
    
    return pred_flows, pred_occs

UNIMATCH_WEIGHTS = {
    'sintel': 'unimatch_weights/gmflow-scale2-regrefine6-sintelft-6e39e2b9.pth',
    'kitti': 'unimatch_weights/gmflow-scale2-regrefine6-kitti15-25b554d7.pth',
    'mix': 'unimatch_weights/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth',
}

