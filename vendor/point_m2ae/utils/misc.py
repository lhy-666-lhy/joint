"""Minimal helpers used by Point-M2AE modules (FPS via pointnet2_ops)."""

from pointnet2_ops import pointnet2_utils


def fps(data, number):
    """Furthest point sampling. data: (B, N, 3) -> (B, number, 3)."""
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = (
        pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx)
        .transpose(1, 2)
        .contiguous()
    )
    return fps_data
