# camera_utils.py
# 작동하는 카메라를 자동으로 찾아주는 공용 유틸리티
#
# 사용법:
#   from camera_utils import find_camera
#   CAMERA_ID = find_camera()

import cv2


def find_camera(max_index: int = 5) -> int:
    """
    0번부터 순서대로 시도해서 실제로 프레임을 읽을 수 있는
    첫 번째 카메라 번호를 반환한다.

    Args:
        max_index: 시도할 최대 카메라 번호 (기본 0~4)

    Returns:
        작동하는 카메라 번호 (int)

    Raises:
        RuntimeError: 작동하는 카메라가 없을 때
    """
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print(f"[camera] /dev/video{i} 사용")
                return i
        cap.release()

    raise RuntimeError(
        "작동하는 카메라를 찾지 못했습니다.\n"
        "  확인: v4l2-ctl --list-devices\n"
        "  확인: ls /dev/video*"
    )
