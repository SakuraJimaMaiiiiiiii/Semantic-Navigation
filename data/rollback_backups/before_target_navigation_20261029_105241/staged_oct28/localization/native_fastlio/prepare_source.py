"""Pinned upstream algorithm sources; generate a ROS/PCL-free transport adapter copy."""
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
COMMIT = '7cc4175de6f8ba2edf34bab02a42195b141027e9'
IKD_COMMIT = 'e2e3f4e9d3b95a9e66b1ba83dc98d4a05ed8a3c4'


def prepare():
    source = ROOT / 'third_party/FAST_LIO'
    if not source.exists():
        subprocess.run(['git', 'clone', '--recursive', 'https://github.com/hku-mars/FAST_LIO.git', str(source)], check=True)
    subprocess.run(['git', '-C', str(source), 'config', 'core.filemode', 'false'], check=True)
    head = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'], text=True).strip()
    if head != COMMIT:
        subprocess.run(['git', '-C', str(source), 'checkout', '--detach', COMMIT], check=True)
    subprocess.run(['git', '-C', str(source), 'submodule', 'update', '--init'], check=True)
    actual = subprocess.check_output(['git','-C',str(source/'include/ikd-Tree'),'rev-parse','HEAD'], text=True).strip()
    if actual != IKD_COMMIT:
        raise RuntimeError('Unexpected ikd-Tree revision')
    out = ROOT / 'generated'
    out.mkdir(exist_ok=True)
    shutil.copytree(source/'include/IKFoM_toolkit', out/'IKFoM_toolkit', dirs_exist_ok=True)
    for name in ['use-ikfom.hpp', 'so3_math.h']:
        shutil.copy2(source/'include'/name, out/name)
    imu = (source/'src/IMU_Processing.hpp').read_text(encoding='utf-8')
    # Original IMU integration/deskew, replacing message/container declarations only.
    imu = '\n'.join(line for line in imu.splitlines() if not line.startswith('#include'))
    imu = '#include "transport.hpp"\n#include "use-ikfom.hpp"\n' + imu
    imu = imu.replace('double last_lidar_end_time_;', 'double last_lidar_end_time_ = 0.;')
    imu = imu.replace('last_imu_   = meas.imu.back();', 'last_imu_   = meas.imu.back();\n    last_lidar_end_time_ = meas.lidar_end_time;')
    imu = imu.replace('fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);', '// Logging is owned by the Python session.')
    (out/'IMU_Processing.hpp').write_text(imu)
    common = (source/'include/common_lib.h').read_text(encoding='utf-8')
    pose = common[common.index('template<typename T>\nauto set_pose6d'):common.index('/* comment\nplane equation:')]
    plane = common[common.rfind('template<typename T>'):common.rfind('#endif')]
    (out/'geometry.hpp').write_text('// Extracted from pinned FAST_LIO common_lib.h; see upstream LICENSE.\n'+pose+plane)
    for name in ['ikd_Tree.h','ikd_Tree.cpp']:
        text = (source/'include/ikd-Tree'/name).read_text(encoding='utf-8')
        text = text.replace('#include <pthread.h>', '#include "thread_compat.hpp"')
        text = text.replace('#include <unistd.h>', '')
        text = text.replace('#include <pcl/point_types.h>', '#include "point_types.hpp"')
        text = text.replace('template class KD_TREE<pcl::PointXYZ>;', '')
        text = text.replace('template class KD_TREE<pcl::PointXYZI>;', '')
        text = text.replace('pcl::PointXYZINormal', 'PointType')
        (out/name).write_text(text)
    print('Prepared official FAST-LIO', COMMIT)


if __name__ == '__main__':
    prepare()
