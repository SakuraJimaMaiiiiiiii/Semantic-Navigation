"""Reliable organized-depth support for semantic objects, in optical coordinates."""
import numpy as np


def reliable_surface_mask(depth, mask, intrinsics, max_incidence_degrees):
    """Reject depth edges and grazing surfaces using local 3-D surface normals.

    Incidence is measured against the pixel's viewing ray, not the optical axis.
    Invalid/discontinuous neighbors cannot provide a trustworthy normal.
    """
    z = np.asarray(depth, dtype=float)
    result = np.zeros(z.shape, dtype=bool)
    rows, cols = np.nonzero(mask)
    inside = (rows > 0) & (rows < z.shape[0]-1) & (cols > 0) & (cols < z.shape[1]-1)
    rows, cols = rows[inside], cols[inside]
    center = z[rows, cols]
    neighbors = np.array([z[rows, cols-1], z[rows, cols+1],
                          z[rows-1, cols], z[rows+1, cols]])
    valid = np.all(np.isfinite(neighbors) & (neighbors > 0), axis=0)
    valid &= np.all(np.abs(neighbors-center) <= .10+.02*center, axis=0)
    rows, cols, center, neighbors = rows[valid], cols[valid], center[valid], neighbors[:, valid]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    def points(u, v, d):
        return np.column_stack(((u-cx)*d/fx, (v-cy)*d/fy, d))
    du = points(cols+1, rows, neighbors[1])-points(cols-1, rows, neighbors[0])
    dv = points(cols, rows+1, neighbors[3])-points(cols, rows-1, neighbors[2])
    normals = np.cross(du, dv)
    rays = points(cols, rows, center)
    denominator = np.linalg.norm(normals, axis=1)*np.linalg.norm(rays, axis=1)
    cosine = np.divide(np.abs(np.einsum('ij,ij->i', normals, rays)), denominator,
                       out=np.zeros(len(rows)), where=denominator > 1e-12)
    keep = cosine >= np.cos(np.deg2rad(max_incidence_degrees))
    result[rows[keep], cols[keep]] = True
    return result


def central_support(mask, ratio, min_points):
    """Central ROI for the position estimate; full support still defines extent.

    Thin signs/pillars with few interior pixels retain their already filtered
    support. Large masks with an empty/occluded center are not filled from edges.
    """
    rows, cols = np.nonzero(mask)
    if len(rows) <= 2*min_points:
        return mask.copy()
    middle = np.array([(rows.min()+rows.max())*.5, (cols.min()+cols.max())*.5])
    half = ratio*np.array([rows.max()-rows.min()+1, cols.max()-cols.min()+1])*.5
    keep = (np.abs(rows-middle[0]) <= half[0]) & (np.abs(cols-middle[1]) <= half[1])
    result = np.zeros_like(mask)
    result[rows[keep], cols[keep]] = True
    return result
