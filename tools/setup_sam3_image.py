"""Prepare pinned official SAM3 image-only source without replacing core packages."""
import argparse
import ast
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '.tools' / 'sam3'
COMMIT = '7f473cfd983d5eb43040ff608f66bff0d7bdcb53'
# Image text inference imports only these extra dependencies; video extras are not installed.
EXTRAS = {'huggingface-hub': '0.36.0', 'iopath': '0.1.10', 'einops': '0.8.1', 'portalocker': '3.2.0',
          'timm': '1.0.22', 'safetensors': '0.6.2', 'hf-xet': '1.2.0'}
if sys.platform == 'win32':
    EXTRAS['pywin32'] = '311'

PROTECTED = ('torch', 'torchvision', 'numpy', 'opencv-python', 'ftfy', 'ultralytics')


def versions():
    # Preserve duplicate metadata too; do not silently choose one distribution.
    result = {name: [] for name in PROTECTED}
    for dist in importlib.metadata.distributions():
        name = dist.metadata['Name'].lower().replace('_','-')
        if name in result:
            result[name].append(dist.version)
    return {name: sorted(values) for name,values in result.items()}


def lazy_builder(source):
    tree = ast.parse(source)
    prefixes = ('sam3.model.sam1_task_predictor', 'sam3.model.sam3_tracking_predictor',
                'sam3.model.sam3_video_inference', 'sam3.model.sam3_video_predictor',
                'sam3.model.video_tracking_multiplex')
    imports = {}
    edits = []
    lines = source.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in prefixes:
            for alias in node.names:
                imports[alias.asname or alias.name] = ''.join(lines[node.lineno-1:node.end_lineno]).strip()
            edits.append((node.lineno-1, node.end_lineno, ''))
    if len(imports) != 5:
        raise RuntimeError('Unexpected upstream builder imports; review pinned source')
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            used = {n.id for statement in node.body for n in ast.walk(statement) if isinstance(n, ast.Name)}
            if node.name == 'build_sam3_image_model':
                used.discard('SAM3InteractiveImagePredictor')
            required = sorted({imports[name] for name in used & imports.keys()})
            if required:
                first = node.body[0]
                after_doc = isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
                line = first.end_lineno if after_doc else first.lineno-1
                edits.append((line, line, ''.join('    '+s+'\n' for s in required)))
    for start,end,replacement in sorted(edits, reverse=True):
        lines[start:end] = [replacement]
    result = 'from __future__ import annotations\n'+''.join(lines)
    return result.replace('    if enable_inst_interactivity:\n        sam3_pvs_base',
        '    if enable_inst_interactivity:\n        from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor\n        sam3_pvs_base')


def patch_source(root):
    marker = root/'image_only_compat.json'
    if marker.exists():
        if json.loads(marker.read_text())['upstream_commit'] != COMMIT:
            raise RuntimeError('Existing compatibility source has a different version')
        return
    actual = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'], text=True).strip()
    if actual != COMMIT:
        raise RuntimeError(f'Expected upstream {COMMIT}, got {actual}; no files changed')
    if subprocess.check_output(['git','-C',str(root),'status','--porcelain'], text=True).strip():
        raise RuntimeError('Official source has local modifications; refusing to overwrite them')
    builder = root/'sam3/model_builder.py'
    image = root/'sam3/model/sam3_image.py'
    nms = root/'sam3/perflib/nms.py'
    changes = {builder: lazy_builder(builder.read_text())}
    value = image.read_text()
    value = value.replace('from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor',
        'from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor')
    value = value.replace('from sam3.train.data.collator import BatchedDatapoint',
                          'from sam3.model.data_misc import BatchedDatapoint')
    changes[image] = 'from __future__ import annotations\n'+value
    # Use upstream's own exact greedy CPU implementation, returning indices to
    # the original device. No mock Triton module, altered mask score or weights.
    value = nms.read_text()
    start = value.index('    if ious.is_cuda:\n')
    end = value.index('    return generic_nms_cpu', start)
    changes[nms] = value[:start]+value[end:]
    for path,content in changes.items():
        compile(content,str(path),'exec')
    for path,content in changes.items():
        path.write_text(content,encoding='utf-8')
    marker.write_text(json.dumps(dict(upstream_commit=COMMIT, mode='image_text_only',
        changes=[str(p.relative_to(root)) for p in changes],
        nms='upstream CPU implementation; preserves output device'),indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-only',action='store_true')
    args=parser.parse_args()
    if not SOURCE.exists():
        subprocess.run(['git','clone','https://github.com/facebookresearch/sam3',str(SOURCE)],check=True)
        subprocess.run(['git','-C',str(SOURCE),'checkout',COMMIT],check=True)
    patch_source(SOURCE)
    if args.source_only:
        print(f'Image-only official source prepared: {SOURCE}')
        return
    before=versions()
    if any(not before[k] for k in ('torch','torchvision','numpy')):
        raise RuntimeError('Activate the existing navigation environment first')
    for name,version in EXTRAS.items():
        try:
            installed=importlib.metadata.version(name)
            print(f'Keeping {name}=={installed}')
        except importlib.metadata.PackageNotFoundError:
            subprocess.run([sys.executable,'-m','pip','install','--no-deps',f'{name}=={version}'],check=True)
    after=versions()
    if before != after:
        raise RuntimeError(f'Protected packages changed unexpectedly: {before} -> {after}')
    (SOURCE/'image_only_environment.json').write_text(json.dumps(dict(
        python=sys.version, executable=sys.executable, protected_before=before, protected_after=after,
        extras={name:importlib.metadata.version(name) for name in EXTRAS}),indent=2),encoding='utf-8')
    print('Prepared image-only source; protected packages unchanged. Run tools/sam3_service.py --check-only next.')


if __name__ == '__main__':
    main()
