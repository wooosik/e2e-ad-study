# e2e-ad-study

E2E Autonomous Driving Study

This repository contains code snippets and configuration files for the
E2E Autonomous Driving study plan.

| Week | 세부 목표 | Recommended To-Dos (✔ 체크리스트) | Week-end Deliverable |
| ---- | -------- | -------------------------------- | ------------------- |
| **0 Kick-off** | 개발 환경 구축 | - Anaconda + PyTorch 2.x CUDA 설치 및 GPU 확인<br>- VS Code / Jupyter 설정 (Black, isort, nb-strip-out)<br>- GitHub Repo `e2e-av-study` 개설 | ✅ `environment.yml`, 첫 커밋 |
| **1. Python + PyTorch** | Tensor 연산·Autograd | 1. PyTorch “Learn the Basics” 30 분 튜토리얼 완주<br>2. MNIST → 2-layer MLP 73 줄로 구현<br>3. `torchviz`로 Computational Graph 시각화 | 노트북 `01_basics.ipynb`<br>MLP 99 % acc 스크린샷 |
| **2. 딥러닝 기초** | CNN·RNN·Optimizer | 1. MIT 6.S191 2024 Lecture 1 & 2 시청 + 노트 정리<br>2. CIFAR-10 CNN 재현(NVIDIA 2016 E2E Net)<br>3. LR/Batch Size sweep → TensorBoard 로깅 | `02_dl_core.ipynb`<br>스칼라 로그 + 커브 비교 |
| **3. 자율주행 개론** | 시스템 & 센서 | 1. Udacity S-D Car ND Behavioral Cloning 프로젝트 README 정독<br>2. 자율주행 스택(Perception-Planning-Control) 요약 1 p 노트 작성<br>3. OpenCV → PIL → `torchvision.transforms` 로 **RGB→Tensor** 파이프라인 실습 | 노트북 `03_pipeline.ipynb` (프레임 → Tensor 변환 및 시각화) |
| **4. 수학 & 차량 모델** | Frenet·Bicycle | 1. GitHub `Kinematic_Bicycle_Model.ipynb` 코드 따라치며 주행 궤적 시뮬레이션<br>2. YouTube “Frenet Frames for Motion Planning” 시청 후, Cartesian→Frenet 좌표 변환 함수 구현<br>3. MathWorks 예제 참고해 고속도로 Trajectory plan 개념 메모 | `04_vehicle_math.ipynb`<br> • Bicycle model 시뮬 gif<br> • Frenet s-d 좌표 변환 결과표 |

## Week 0: Kick-off

Create the conda environment and install pre-commit hooks for automatic
formatting. Then verify that PyTorch can access your GPU.

```bash
conda env create -f environment.yml
conda activate e2e-ad-study
pre-commit install
```

Verify that PyTorch sees your GPU by running:

```bash
python check_gpu.py
```

## Week 1: Python + PyTorch

Open `01_basics.ipynb` in Jupyter. The notebook
walks through tensor operations and trains a simple 2-layer MLP on MNIST.
It also shows how to visualize the network's computational graph using
`torchviz`. Train for a few epochs until you reach about 99% accuracy,
then capture a screenshot for your records.

## Week 2: Deep Learning Core

Launch `02_dl_core.ipynb` to reproduce the NVIDIA E2E network for
CIFAR-10. Use the provided sweep function to compare scalar curves for
different learning rates and batch sizes in TensorBoard.

## Week 3: 자율주행 개론

Read the Udacity Behavioral Cloning project README and prepare a
one‑page summary of the autonomous driving stack
(Perception→Planning→Control). Use `03_pipeline.ipynb` to practice
loading a frame with OpenCV, converting it to PIL, and transforming it to
a tensor with `torchvision`. Visualize the tensor to confirm the
conversion.

`data_pipeline.py` also demonstrates how to load an openpilot segment. The
`load_segment_data()` function reads the camera video, steering logs and
global position data so that it can be fed into a model like
`SupercomboModel`.

## Week 4: 수학 & 차량 모델

`study_plan.py` contains minimal examples for the later weeks of the
plan. Set the environment variable `SAMPLE_IMAGE` to an RGB image path to
try the frame‑to‑tensor conversion. The script also includes a simple
kinematic bicycle model step for motion‑planning experiments.
