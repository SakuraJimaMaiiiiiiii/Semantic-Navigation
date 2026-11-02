"""Offline evaluation against optional same-frame NED object ground truth."""

import numpy as np


def evaluate_map(records, truth, match_distance=2.0):
    """Nearest same-class GT association; duplicate predictions counted separately."""
    matches, errors, assignments = {}, [], {}
    for record in records:
        candidates = [g for g in truth if g["class_name"] == record["class_name"]]
        if not candidates:
            continue
        best = min(
            candidates,
            key=lambda g: np.linalg.norm(
                np.array(g["position_world"]) - record["position_world"]
            ),
        )
        error = float(
            np.linalg.norm(np.array(best["position_world"]) - record["position_world"])
        )
        if error > match_distance:
            continue
        identity = best["instance_id"]
        matches.setdefault(identity, []).append(error)
        assignments[record.get("runtime_instance_id", record["instance_id"])] = identity
    errors = [min(values) for values in matches.values()]
    duplicates = sum(len(values) - 1 for values in matches.values())
    return {
        "ground_truth_count": len(truth),
        "prediction_count": len(records),
        "matched_ground_truth_count": len(matches),
        "duplicate_count": duplicates,
        "duplicate_rate": duplicates / len(records) if records else 0.0,
        "recall": len(matches) / len(truth) if truth else None,
        "unmatched_prediction_count": len(records) - len(assignments),
        "position_rmse_m": (
            float(np.sqrt(np.mean(np.square(errors)))) if errors else None
        ),
        "associations": assignments,
    }


def evaluate_identity_frames(frames, truth, match_distance=2.0):
    previous, switches, timestamps = {}, 0, set()
    for frame in sorted(frames, key=lambda f: f["timestamp"]):
        if frame["timestamp"] in timestamps:
            continue
        timestamps.add(frame["timestamp"])
        # Only fresh observations, not stale map memories, contribute to ID switches.
        records = [
            r
            for r in frame["records"]
            if abs(r["last_seen"] - frame["timestamp"]) < 1e-6
        ]
        associations = evaluate_map(records, truth, match_distance)["associations"]
        grouped = {}
        for runtime_id, gt_id in associations.items():
            grouped.setdefault(gt_id, []).append(runtime_id)
        for gt_id, ids in grouped.items():
            if (
                len(ids) != 1
            ):  # Duplicates are measured separately, not arbitrary switches.
                continue
            if gt_id in previous and previous[gt_id] != ids[0]:
                switches += 1
            previous[gt_id] = ids[0]
    return {"id_switches": switches, "frame_count": len(timestamps)}


def summarize_runs(results):
    count = len(results)
    return {
        "run_count": count,
        "success_rate": (
            sum(bool(r["success"]) for r in results) / count if count else None
        ),
        "collision_rate": (
            sum(bool(r["collision"]) for r in results) / count if count else None
        ),
        "mean_elapsed_seconds": (
            float(np.mean([r["elapsed_seconds"] for r in results])) if count else None
        ),
        "mean_path_length_m": (
            float(np.mean([r["path_length_m"] for r in results])) if count else None
        ),
        "failure_reasons": {
            reason: sum(r["reason"] == reason for r in results)
            for reason in sorted({r["reason"] for r in results if not r["success"]})
        },
    }
