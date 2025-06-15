import cv2
import numpy as np


def load_segment_data(segment_path: str):
    """Load image frames, steering logs, and positions from a data segment."""
    # 1. 영상 로드 (HEVC 파일 -> 프레임 시퀀스)
    video_path = f"{segment_path}/video.hevc"
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 프레임 전처리: 해상도 변환 및 색공간 YUV 변환
        frame_resized = cv2.resize(frame, (256, 128))
        frame_yuv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2YUV)
        frames.append(frame_yuv)
    cap.release()
    frames = np.array(frames)  # shape: (N, 128, 256, 3)

    # 2. 로그 로드 (스티어링 각도, 속도 등)
    steer_log = np.load(f"{segment_path}/processed_log/steering_angle/value.npy")
    time_log = np.load(f"{segment_path}/processed_log/steering_angle/t.npy")

    # 3. 전역 위치 로드 (미래 궤적 생성용)
    xyz_global = np.load(f"{segment_path}/global_pos/global_positions.npy")

    return frames, steer_log, xyz_global
