import ast
import unittest
from tools.setup_sam3_image import lazy_builder


class ImageCompatibilityTests(unittest.TestCase):
    def test_image_builder_does_not_eagerly_import_optional_interactivity(self):
        source = '''from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor
from sam3.model.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity
from sam3.model.sam3_video_predictor import Sam3VideoPredictorMultiGPU
from sam3.model.video_tracking_multiplex import VideoTrackingDynamicMultiplex

def build_sam3_image_model(enable_inst_interactivity=False):
    if enable_inst_interactivity:
        sam3_pvs_base = build_tracker()
        return SAM3InteractiveImagePredictor(sam3_pvs_base)
    return 'image-only'

def build_tracker() -> Sam3TrackerPredictor:
    return Sam3TrackerPredictor()
'''
        patched = lazy_builder(source)
        namespace = {}
        # No sam3/triton modules exist in the test process. False branch must
        # work without trying any video import; annotations must be deferred.
        exec(compile(patched,'<compat>','exec'),namespace)
        self.assertEqual(namespace['build_sam3_image_model'](), 'image-only')
        tree = ast.parse(patched)
        self.assertFalse(any(isinstance(n,ast.ImportFrom) and (n.module or '').startswith('sam3')
                             for n in tree.body))

    def test_unknown_upstream_import_layout_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'Unexpected upstream'):
            lazy_builder('def build_sam3_image_model(): pass\n')
