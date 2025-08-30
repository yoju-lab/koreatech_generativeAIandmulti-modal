#!/bin/bash
# RunPod A40에서 MiniGPT-4 실행을 위한 설치 스크립트
# 기본 환경: Ubuntu + Python3 + A40 GPU

echo "🚀 MiniGPT-4 설치 시작..."

# ✅ pip 기반 필수 패키지 설치
pip install -r minigpt-4_requirements.txt

# ✅ MiniGPT-4 저장소 클론 및 모델 가중치 다운로드
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
mkdir -p checkpoints
wget https://huggingface.co/wangrongsheng/MiniGPT4-7B/resolve/main/prerained_minigpt4_7b.pth -P checkpoints/

# ✅ MiniGPT-4 실행 (Gradio 웹 데모)
# rm -rf ~/.cache/huggingface/hub/models--TheBloke--vicuna-7B-1.1-HF
python demo.py --cfg-path /workspace/minigpt-4_eval_public_vicuna.yaml --gpu-id 0


