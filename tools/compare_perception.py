"""Compare front ends on deterministic saved BGR frames; never command a UAV."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def frames(path, step, count):
    if path.suffix.lower() in {'.h5', '.hdf5'}:
        import h5py
        with h5py.File(path, 'r') as data:
            for index in range(0, min(len(data['sensors/rgb']), step*count), step):
                yield index, np.asarray(data['sensors/rgb'][index])
    else:
        video = cv2.VideoCapture(str(path))
        if not video.isOpened():
            raise ValueError(f'Cannot open {path}')
        try:
            for index in range(step*count):
                ok, frame = video.read()
                if not ok:
                    break
                if index % step == 0:
                    yield index, frame
        finally:
            video.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Recorded H5 or video')
    parser.add_argument('--backend', choices=['yolo_sam2', 'sam3'], required=True)
    parser.add_argument('--sam3-url', default='http://127.0.0.1:8878')
    parser.add_argument('--object-prompts', nargs='+', default=['car', 'fire hose cabinet', 'fire extinguisher'])
    parser.add_argument('--step', type=int, default=100)
    parser.add_argument('--frames', type=int, default=30)
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing output is never overwritten')
    args = parser.parse_args()
    if args.step < 1 or args.frames < 1:
        parser.error('step and frames must be positive')
    from detections.object_detect import DetectorConfig, TiledVehicleDetector, SAM2Segmenter, semantic_class_color
    config = replace(DetectorConfig(), backend=args.backend, sam3_url=args.sam3_url,
                     class_prompts=tuple(args.object_prompts), semantic_landmark_classes=tuple(args.object_prompts))
    if args.backend == 'sam3':
        from detections.sam3_client import SAM3Detector
        detector, segmenter = SAM3Detector(config), None
    else:
        detector, segmenter = TiledVehicleDetector(config), SAM2Segmenter(config)
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for index, image in frames(args.input, args.step, args.frames):
        started = time.perf_counter()
        detections = detector.detect(image)
        detected = time.perf_counter()
        if segmenter is None:
            active, masks = list(enumerate(detections)), detector.masks
        elif detections:
            active = segmenter._select_sam_detections(detections)
            masks = segmenter._predict_masks(image, active)
            masks = [] if masks is None else masks
        else:
            active, masks = [], []
        finished = time.perf_counter()
        overlay = image.copy()
        saved_masks, mask_indices = [], []
        for (detection_index, detection), mask in zip(active, masks):
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(np.asarray(mask, dtype=np.float32), image.shape[1::-1], interpolation=cv2.INTER_NEAREST)
            foreground = mask > .5
            overlay[foreground] = semantic_class_color(detection.class_name)
            saved_masks.append(foreground)
            mask_indices.append(detection_index)
        preview = detector.draw(cv2.addWeighted(image, .6, overlay, .4, 0.), detections)
        cv2.imwrite(str(args.output/f'frame_{index:06d}.jpg'), preview)
        np.savez_compressed(args.output/f'frame_{index:06d}.npz',
                            masks=np.asarray(saved_masks,dtype=bool).reshape(-1,*image.shape[:2]),
                            detection_indices=np.asarray(mask_indices,dtype=int))
        results.append(dict(frame_index=index, detections=[asdict(d) for d in detections],
                            mask_count=len(saved_masks), detector_seconds=detected-started,
                            mask_seconds=finished-detected, total_seconds=finished-started))
        print(f'{args.backend}: frame={index}, detections={len(detections)}, masks={len(saved_masks)}, seconds={finished-started:.3f}')
    if not results:
        raise ValueError('No frames decoded')
    seconds = [r['total_seconds'] for r in results]
    report = dict(backend=args.backend, input=str(args.input.resolve()), prompts=args.object_prompts,
                  config=asdict(config), service=getattr(detector,'metadata',None),
                  mean_seconds=float(np.mean(seconds)), p95_seconds=float(np.percentile(seconds,95)),
                  mean_seconds_excluding_first=float(np.mean(seconds[1:] or seconds)),
                  note='Raw visual comparison, not semantic map replay or ground-truth accuracy.', frames=results)
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


if __name__ == '__main__':
    main()
