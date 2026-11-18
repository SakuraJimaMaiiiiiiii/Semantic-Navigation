"""Create an explicitly SYNTHETIC occupancy archive for offline graph validation."""
import argparse
from pathlib import Path

import numpy as np

from planner.vertical_projection import height_band_indices


def make_demo(path, scenario):
    """Same archive schema as global occupancy; geometry is entirely synthetic."""
    path = Path(path)
    resolution, altitude, half_extent = .25, -1.5, .25
    area = {(x, y) for x in range(-32, 33) for y in range(-20, 21)}
    walls = {c for c in area if abs(c[0]) == 32 or abs(c[1]) == 20}
    if scenario == 'wall':
        walls.update(c for c in area if c[0] == 0)
    elif scenario == 'door':
        walls.update(c for c in area if c[0] == 0 and abs(c[1]) > 5)
        # Parked vehicle-sized block and an independent column.
        walls.update((x, y) for x in range(9, 18) for y in range(-13, -5))
        walls.update((x, y) for x in range(-14, -11) for y in range(8, 11))
    elif scenario == 'dead_end':
        walls.update((x, y) for x in range(-8, 17) for y in (-6, 6))
        walls.update((16, y) for y in range(-6, 7))
    else:
        raise ValueError('Unknown synthetic scenario')
    low, high = height_band_indices(altitude, half_extent, resolution)
    indices = np.asarray([(x, y, z) for x, y in sorted(area) for z in range(low, high+1)], dtype=np.int32)
    odds = np.asarray([2. if (int(x), int(y)) in walls else -2. for x, y, _ in indices], dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, indices=indices, log_odds=odds, resolution=resolution,
                            timestamp=0., free_threshold=-.5, occupied_threshold=.5,
                            dataset_label=f'SYNTHETIC {scenario}; not a sensor map',
                            recommended_altitude_ned=altitude)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=['door', 'wall', 'dead_end'], default='door')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        make_demo(args.output, args.scenario)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f'SYNTHETIC input: {args.output}')


if __name__ == '__main__':
    main()
