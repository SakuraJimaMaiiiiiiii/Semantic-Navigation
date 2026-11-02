"""Export an occupied PLY as Foxglove point-cloud and solid voxel-grid topics.

This is an offline visualization tool. It does not change the mapping, planning,
or flight-control data used by the demo.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import struct
import time

import h5py
import numpy as np
import foxglove
from foxglove.channels import (
    FrameTransformChannel,
    PointCloudChannel,
    VoxelGridChannel,
)
from foxglove.messages import (
    FrameTransform,
    PackedElementField,
    PackedElementFieldNumericType,
    PointCloud,
    Pose,
    Quaternion,
    Timestamp,
    Vector3,
    VoxelGrid,
)


RGBA_FIELDS = [
    PackedElementField(name="red", offset=0, type=PackedElementFieldNumericType.Uint8),
    PackedElementField(name="green", offset=1, type=PackedElementFieldNumericType.Uint8),
    PackedElementField(name="blue", offset=2, type=PackedElementFieldNumericType.Uint8),
    PackedElementField(name="alpha", offset=3, type=PackedElementFieldNumericType.Uint8),
]


def read_binary_occupied_ply(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    marker = b"end_header\n"
    header_end = payload.index(marker) + len(marker)
    header = payload[:header_end].decode("ascii")
    count_line = next(
        line for line in header.splitlines() if line.startswith("element vertex ")
    )
    count = int(count_line.split()[-1])
    source_stride = 19  # xyz float32 + rgb uint8 + log_odds float32
    points = np.empty((count, 3), dtype=np.float32)
    for index in range(count):
        points[index] = struct.unpack_from(
            "<fff", payload, header_end + index * source_stride
        )
    return points


def height_colors(points: np.ndarray, alpha: int = 235) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 4), dtype=np.uint8)
    z = points[:, 2].astype(np.float64)
    ratio = np.clip(
        (z - float(np.min(z))) / max(float(np.ptp(z)), 1e-6), 0.0, 1.0
    )
    colors = np.empty((len(points), 4), dtype=np.uint8)
    colors[:, 0] = np.clip(35 + 210 * ratio, 0, 255).astype(np.uint8)
    colors[:, 1] = np.clip(175 - 80 * ratio, 0, 255).astype(np.uint8)
    colors[:, 2] = np.clip(240 - 170 * ratio, 0, 255).astype(np.uint8)
    colors[:, 3] = np.uint8(alpha)
    return colors


def solid_colors(count: int) -> np.ndarray:
    return np.tile(np.asarray([255, 145, 35, 255], dtype=np.uint8), (count, 1))


def point_cloud(points: np.ndarray, colors: np.ndarray, stamp: Timestamp) -> PointCloud:
    vertices = np.empty(
        len(points),
        dtype=np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("alpha", "u1"),
        ]),
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"], vertices["alpha"] = colors.T
    fields = [
        PackedElementField(name="x", offset=0, type=PackedElementFieldNumericType.Float32),
        PackedElementField(name="y", offset=4, type=PackedElementFieldNumericType.Float32),
        PackedElementField(name="z", offset=8, type=PackedElementFieldNumericType.Float32),
        PackedElementField(name="red", offset=12, type=PackedElementFieldNumericType.Uint8),
        PackedElementField(name="green", offset=13, type=PackedElementFieldNumericType.Uint8),
        PackedElementField(name="blue", offset=14, type=PackedElementFieldNumericType.Uint8),
        PackedElementField(name="alpha", offset=15, type=PackedElementFieldNumericType.Uint8),
    ]
    return PointCloud(
        timestamp=stamp,
        frame_id="map",
        pose=Pose(position=Vector3(), orientation=Quaternion(w=1.0)),
        point_stride=16,
        fields=fields,
        data=vertices.tobytes(),
    )


def voxel_grid(
    points: np.ndarray,
    colors: np.ndarray,
    resolution: float,
    stamp: Timestamp,
) -> VoxelGrid:
    indices = np.floor(points.astype(np.float64) / float(resolution)).astype(np.int64)
    minimum = indices.min(axis=0)
    maximum = indices.max(axis=0)
    size = maximum - minimum + 1
    nx, ny, nz = map(int, size)
    data = np.zeros((nz, ny, nx, 4), dtype=np.uint8)
    local = indices - minimum
    data[local[:, 2], local[:, 1], local[:, 0]] = colors
    cell_stride = 4
    row_stride = nx * cell_stride
    slice_stride = ny * row_stride
    origin = minimum.astype(np.float64) * float(resolution)
    return VoxelGrid(
        timestamp=stamp,
        frame_id="map",
        pose=Pose(
            position=Vector3(x=float(origin[0]), y=float(origin[1]), z=float(origin[2])),
            orientation=Quaternion(w=1.0),
        ),
        row_count=ny,
        column_count=nx,
        cell_size=Vector3(x=resolution, y=resolution, z=resolution),
        slice_stride=slice_stride,
        row_stride=row_stride,
        cell_stride=cell_stride,
        fields=RGBA_FIELDS,
        data=data.tobytes(),
    )


def export(
    ply_path: Path,
    h5_path: Path,
    output_path: Path,
    *,
    resolution: float | None = None,
    drone_down: float | None = None,
) -> Path:
    points = read_binary_occupied_ply(ply_path)
    if resolution is None or drone_down is None:
        with h5py.File(h5_path, "r") as h5_file:
            resolution = float(h5_file.attrs["resolution"])
            drone_down = float(
                np.asarray(h5_file["drone_position_ned"])[2]
            )
    resolution = float(resolution)
    drone_down = float(drone_down)

    flight_up = -drone_down
    slice_min_up = flight_up - 0.9
    slice_max_up = flight_up + 0.25
    slice_mask = (points[:, 2] >= slice_min_up) & (points[:, 2] <= slice_max_up)
    navigation_points = points[slice_mask]

    now_ns = time.time_ns()
    stamp = Timestamp(now_ns // 1_000_000_000, now_ns % 1_000_000_000)
    full_colors = height_colors(points)
    navigation_colors = solid_colors(len(navigation_points))

    channels = {
        "transform": FrameTransformChannel("/tf"),
        "full_points": PointCloudChannel("/garage/occupied_all"),
        "slice_points": PointCloudChannel("/garage/navigation_slice"),
        "full_voxels": VoxelGridChannel("/garage/occupied_voxels"),
        "slice_voxels": VoxelGridChannel("/garage/navigation_voxels"),
    }
    with foxglove.open_mcap(str(output_path)):
        # Register ``map`` in Foxglove's transform tree.  Without at least one
        # FrameTransform, the 3D panel marks the fixed/display frames invalid
        # even when PointCloud and VoxelGrid messages use frame_id="map".
        channels["transform"].log(
            FrameTransform(
                timestamp=stamp,
                parent_frame_id="map",
                child_frame_id="base_link",
                translation=Vector3(),
                rotation=Quaternion(w=1.0),
            )
        )
        channels["full_points"].log(point_cloud(points, full_colors, stamp))
        channels["slice_points"].log(
            point_cloud(navigation_points, navigation_colors, stamp)
        )
        if len(points):
            channels["full_voxels"].log(
                voxel_grid(points, full_colors, resolution, stamp)
            )
        if len(navigation_points):
            channels["slice_voxels"].log(
                voxel_grid(
                    navigation_points,
                    navigation_colors,
                    resolution,
                    stamp,
                )
            )

    print(f"Foxglove MCAP: {output_path}")
    print(f"Resolution: {resolution:.3f} m")
    print(f"Full occupied voxels: {len(points)}")
    print(f"Navigation voxels: {len(navigation_points)}")
    print(f"Navigation Up range: [{slice_min_up:.3f}, {slice_max_up:.3f}] m")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("occupied_ply", type=Path)
    parser.add_argument("map_h5", type=Path)
    parser.add_argument("output_mcap", type=Path)
    args = parser.parse_args()
    export(args.occupied_ply, args.map_h5, args.output_mcap)


if __name__ == "__main__":
    main()
