# TensorBoard 사용 가이드

STAIR-RL 학습 과정을 실시간으로 모니터링하기 위한 TensorBoard 사용 가이드입니다.

## 🚀 빠른 시작

### 1. 학습 시작
```bash
# Phase 1: CQL-SAC 학습 (임베딩 자동 로드)
python scripts/run_training.py --phase 1 --steps 500000 --gpu 0

# 또는 Phase 2: PPO-CVaR 학습
python scripts/run_training.py --phase 2 --steps 100000 --gpu 0 \
  --pretrained checkpoints/phase1/cql_sac_final.pt
```

학습이 시작되면 콘솔에 다음과 같은 메시지가 표시됩니다:
```
TensorBoard logging to: checkpoints/run_20250115_143022/phase1/tensorboard
  View with: tensorboard --logdir checkpoints/run_20250115_143022/phase1/tensorboard
```

### 2. TensorBoard 실행

**방법 1: 헬퍼 스크립트 사용 (권장)**
```bash
# 최신 체크포인트 디렉토리를 자동으로 찾아서 실행
./scripts/launch_tensorboard.sh

# 또는 특정 체크포인트 디렉토리 지정
./scripts/launch_tensorboard.sh checkpoints/run_20250115_143022
```

**방법 2: 직접 실행**
```bash
# Phase 1만 모니터링
tensorboard --logdir checkpoints/run_20250115_143022/phase1/tensorboard \
  --port 6006 --bind_all

# Phase 1 + Phase 2 동시 모니터링
tensorboard --logdir_spec \
  phase1:checkpoints/run_20250115_143022/phase1/tensorboard,\
  phase2:checkpoints/run_20250115_143022/phase2/tensorboard \
  --port 6006 --bind_all
```

### 3. 브라우저에서 접속

TensorBoard가 실행되면 다음 주소로 접속:
- **로컬**: http://localhost:6006
- **원격 서버**: http://<서버IP>:6006

---

## 📊 로깅되는 메트릭

### Phase 1: CQL-SAC (Offline Pre-training)

#### Loss Metrics
- **Loss/Critic**: Critic 네트워크 손실 (Q-value prediction error)
- **Loss/Actor**: Actor 네트워크 손실 (policy gradient)
- **Loss/CQL**: Conservative Q-Learning 정규화 손실
- **Loss/Total**: 전체 손실 (Critic + Actor + CQL)

#### SAC Metrics
- **SAC/Alpha**: SAC 온도 파라미터 (entropy coefficient)
- **Q-Value/Q1_Mean**: Q1 네트워크의 평균 Q-value
- **Q-Value/Q2_Mean**: Q2 네트워크의 평균 Q-value (twin Q-networks)

#### Gradient Metrics
- **GradNorm/Actor**: Actor 그래디언트 norm (exploding/vanishing 체크)
- **GradNorm/Critic**: Critic 그래디언트 norm

**해석:**
- `Loss/Critic`가 감소하면 Q-value 예측이 개선됨
- `Loss/CQL`이 안정적이면 보수적 학습이 잘 되는 것
- `SAC/Alpha`가 자동으로 조정되면서 exploration-exploitation 균형 유지
- `GradNorm`이 너무 크거나 작으면 학습이 불안정

---

### Phase 2: PPO-CVaR (Online Fine-tuning)

#### Episode Metrics
- **Episode/Reward**: 에피소드별 누적 리워드 (수익률)
- **Episode/Steps**: 에피소드 길이
- **Episode/TransactionCost**: 에피소드별 총 거래 비용
- **Episode/Turnover**: 에피소드별 포트폴리오 회전율 (churning)

#### PPO Loss Metrics
- **Loss/Policy**: Policy 손실 (clipped surrogate objective)
- **Loss/Value**: Value 손실 (V-function prediction error)
- **Loss/Entropy**: Entropy 손실 (exploration bonus)
- **Loss/Total**: 전체 손실

#### CVaR Metrics
- **CVaR/Value**: 현재 CVaR 값 (95% confidence level)
- **CVaR/Lambda**: CVaR 제약의 라그랑주 승수 (λ)
- **CVaR/Violation**: CVaR 제약 위반 정도

#### Policy Metrics
- **Policy/Entropy**: Policy entropy (높을수록 더 탐험적)
- **Policy/KL_Divergence**: KL divergence (policy 변화량)
- **Policy/ClipFraction**: PPO clipping 비율

**해석:**
- `Episode/Reward`가 증가하면 전략이 개선됨
- `Episode/TransactionCost`가 낮아지면 lazy trading이 잘 작동
- `CVaR/Value`가 낮아지면 리스크가 감소 (목표: κ=5% 이하)
- `CVaR/Lambda`가 증가하면 CVaR 제약이 더 강하게 적용됨
- `Policy/KL_Divergence`가 너무 크면 학습이 불안정 (PPO clip 필요)

---

## 🔍 모니터링 팁

### 1. 학습 진행 확인

**정상적인 학습 패턴:**
- ✅ **Loss가 감소**: Critic Loss, Actor Loss가 시간에 따라 감소
- ✅ **Q-value 안정화**: Q1, Q2가 발산하지 않고 안정적
- ✅ **Reward 증가**: Episode Reward가 평균적으로 증가 추세
- ✅ **CVaR 감소**: CVaR 값이 목표치(5%) 이하로 유지

**문제 신호:**
- ❌ **Loss 폭발**: 손실이 급격히 증가 → learning rate 감소 필요
- ❌ **Q-value 발산**: Q-value가 계속 증가 → CQL 강화 필요
- ❌ **Reward 정체**: 100 episode 이상 개선 없음 → exploration 강화
- ❌ **CVaR 위반**: CVaR이 계속 5% 초과 → λ 증가 필요

### 2. 비교 실험

병렬로 다른 하이퍼파라미터 실험을 실행한 경우:
```bash
# GPU 0: LR 0.0001
CUDA_VISIBLE_DEVICES=0 python scripts/run_training.py --phase 1 --steps 500000 \
  --checkpoint-dir checkpoints/lr_0001 &

# GPU 1: LR 0.0003
CUDA_VISIBLE_DEVICES=1 python scripts/run_training.py --phase 1 --steps 500000 \
  --checkpoint-dir checkpoints/lr_0003 &

# 두 실험을 동시에 모니터링
tensorboard --logdir_spec \
  lr_0001:checkpoints/lr_0001/phase1/tensorboard,\
  lr_0003:checkpoints/lr_0003/phase1/tensorboard \
  --port 6006 --bind_all
```

TensorBoard 좌측 하단의 "Runs" 메뉴에서 실험별로 색상이 다르게 표시됩니다.

### 3. 스무딩(Smoothing) 조정

TensorBoard 좌측 메뉴에서 "Smoothing" 슬라이더를 조정하여:
- **0.0**: 원본 데이터 (노이즈 많음)
- **0.6** (기본값): 적당한 스무딩
- **0.9**: 강한 스무딩 (트렌드만 보임)

### 4. 특정 메트릭 다운로드

TensorBoard 우측 상단의 다운로드 버튼으로 CSV 파일 저장 가능:
- 논문 그래프 작성용
- 추가 분석용

---

## 🎯 학습 목표 및 기준

### Phase 1 (CQL-SAC) 성공 기준
- ✅ `Loss/CQL` < 0.5 (보수적 학습 안정화)
- ✅ `Q-Value/Q1_Mean` 수렴 (발산하지 않음)
- ✅ `GradNorm/Actor`, `GradNorm/Critic` < 10 (안정적 학습)

### Phase 2 (PPO-CVaR) 성공 기준
- ✅ `Episode/Reward` > 0 (평균적으로 수익)
- ✅ `CVaR/Value` < 0.05 (5% 리스크 제약 만족)
- ✅ `Episode/TransactionCost` < 0.01 (1% 이하 거래 비용)
- ✅ `Policy/KL_Divergence` < 0.1 (안정적 policy 업데이트)

---

## 📁 로그 파일 구조

```
checkpoints/
└── run_20250115_143022/
    ├── phase1/
    │   ├── tensorboard/          # TensorBoard 로그
    │   │   └── events.out.tfevents.*
    │   ├── cql_sac_step_50000.pt
    │   ├── cql_sac_step_100000.pt
    │   └── cql_sac_final.pt
    └── phase2/
        ├── tensorboard/          # TensorBoard 로그
        │   └── events.out.tfevents.*
        ├── ppo_cvar_step_20000.pt
        └── ppo_cvar_final.pt
```

---

## 🌐 원격 서버 접속 (SSH 터널링)

원격 서버에서 학습 중인 경우, 로컬 브라우저에서 TensorBoard를 보려면:

```bash
# 로컬 컴퓨터에서 실행 (SSH 포트 포워딩)
ssh -L 6006:localhost:6006 user@remote-server

# 그 다음 원격 서버에서 TensorBoard 실행
tensorboard --logdir checkpoints/run_20250115_143022/phase1/tensorboard

# 로컬 브라우저에서 http://localhost:6006 접속
```

---

## ⚙️ 고급 옵션

### 다른 포트 사용
```bash
tensorboard --logdir checkpoints/run_20250115_143022/phase1/tensorboard \
  --port 6007 --bind_all
```

### 업데이트 주기 조정
```bash
# 30초마다 새로고침 (기본값: 30초)
tensorboard --logdir ... --reload_interval 30
```

### 여러 실험 디렉토리 동시 모니터링
```bash
tensorboard --logdir checkpoints/ --port 6006 --bind_all
```
모든 하위 디렉토리의 tensorboard 로그를 자동으로 감지합니다.

---

## 🐛 문제 해결

### TensorBoard가 실행되지 않을 때
```bash
# TensorBoard 설치 확인
pip install tensorboard

# 포트가 이미 사용 중인 경우
lsof -ti:6006 | xargs kill -9  # 기존 프로세스 종료
```

### 로그가 보이지 않을 때
```bash
# 로그 디렉토리 확인
ls -la checkpoints/run_20250115_143022/phase1/tensorboard/

# events 파일이 있는지 확인
# events.out.tfevents.* 파일이 있어야 함
```

### 그래프가 업데이트되지 않을 때
- 브라우저에서 **Ctrl+Shift+R** (강제 새로고침)
- TensorBoard 재시작

---

## 📚 추가 자료

- [TensorBoard 공식 문서](https://www.tensorflow.org/tensorboard)
- [TensorBoard GitHub](https://github.com/tensorflow/tensorboard)

---

**질문이나 문제가 있으면 이슈를 등록해주세요!**
