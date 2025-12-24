import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import savgol_filter, filtfilt, savgol_coeffs
from model import VirtualColumn

print("--- 🚀 디지털 트윈 연구 통합판: 노화 진단 및 적응형 노이즈 필터링 ---")

# ==========================================
# 1. 상황 설정: 노후화된 공장 및 랜덤 노이즈 발생
# ==========================================
real_aging = 0.65
real_plant = VirtualColumn(N=50, V_total=10.0, Q=1.0)
real_plant.aging_factor = real_aging

# 현실 데이터 생성 (이론적 깨끗한 값)
t_real, y_target_clean, _ = real_plant.run_simulation(t_max=100)

# [TEST] 노이즈 강도를 0.2~0.9 사이에서 랜덤하게 바꿔가며 테스트해보세요.
noise_level = 0.7
noise = np.random.normal(0, noise_level, size=y_target_clean.shape) 
y_target_noisy = y_target_clean + noise 

# ---------------------------------------------------------
# [ADAPTIVE LOGIC] 실시간 노이즈 측정 및 윈도우 자동 설정
# ---------------------------------------------------------
# 1. 베이스라인(0~10분) 구간에서 실제 노이즈 수치(표준편차)를 측정합니다.
baseline_noise = np.std(y_target_noisy[0:100]) 

# 2. 측정된 노이즈에 맞춰 윈도우 길이를 결정합니다. (Mapping Function)
auto_window = int(60 * baseline_noise + 9)
if auto_window % 2 == 0: auto_window += 1  # 무조건 홀수 유지

# 3. 결정된 자동 윈도우로 필터 적용
# SG 필터에 사용되는 수학적 계수(Coefficients)를 먼저 추출합니다.
coeffs = savgol_coeffs(window_length=auto_window, polyorder=3)

# filtfilt를 사용하여 앞뒤로 두 번 필터링해 위상 지연을 0으로 만듭니다.
# filtfilt(계수, 분모계수[보통 1.0], 데이터) 순서입니다.
y_filtered = filtfilt(coeffs, [1.0], y_target_noisy)
# ---------------------------------------------------------

print(f"1. 상황 발생: 노화도({real_aging})")
print(f"   >> 노이즈 감지기: σ={baseline_noise:.3f} 감지 -> 윈도우 {auto_window} 설정")

# ==========================================
# 2. AI 진단 (필터링된 데이터를 보고 정답 추측)
# ==========================================
def calculate_error(guess):
    current_guess = guess[0]
    if current_guess <= 0.0 or current_guess > 2.0:
        return 999999999.0
    sim = VirtualColumn(N=50, V_total=10.0, Q=1.0)
    sim.aging_factor = current_guess
    _, y_sim, _ = sim.run_simulation(t_max=100)
    # AI는 필터링된 데이터를 기준으로 오차를 계산합니다.
    return np.mean(((y_filtered - y_sim)**2) * (y_filtered + 1.0))

print("   >> AI가 적응형 필터를 통해 노화도를 역추적 중...")
search_grid = np.linspace(0.1, 2.0, 20)
best_guess = 1.0
min_err = 999999999.0

for g in search_grid:
    err = calculate_error([g])
    if err < min_err:
        min_err = err
        best_guess = g

res = minimize(calculate_error, [best_guess], method='Nelder-Mead', tol=1e-5)
ai_estimated_aging = res.x[0]

print(f"2. AI 진단 완료: 최종 Aging Factor {ai_estimated_aging:.4f} 추정")

# ==========================================
# 3. 골든타임 예측 및 제어 명령
# ==========================================
ai_model = VirtualColumn(N=50, V_total=10.0, Q=1.0)
ai_model.aging_factor = ai_estimated_aging
t_sim, y_sim, _ = ai_model.run_simulation(t_max=100)

adaptive_margin = 9.0 - (ai_estimated_aging * 5.0)
peak_index = np.argmax(y_sim)
peak_time = t_sim[peak_index]
optimal_cut_point = peak_time - adaptive_margin
print(f"✅ 지능형 제어: 노화도 {ai_estimated_aging:.3f}에 맞춰 마진을 {adaptive_margin:.2f}분으로 자동 조정했습니다.")
# ==========================================
# 4. 결과 시각화 (모든 시각화 요소 유지)
# ==========================================
new_plant = VirtualColumn(N=50, V_total=10.0, Q=1.0)
new_plant.aging_factor = 1.0
t_new, y_new, _ = new_plant.run_simulation(t_max=100)

new_peak_t, new_peak_y = t_new[np.argmax(y_new)], np.max(y_new)
real_peak_t, real_peak_y = t_real[np.argmax(y_target_clean)], np.max(y_target_clean)

plt.figure(figsize=(14, 7))

# (1) 새 제품 상태 (회색 점선)
plt.plot(t_new, y_new, color='gray', linestyle=':', label='Brand New Column (Aging=1.0)', alpha=0.5)

# (2) 노이즈 섞인 원본 (매우 연한 빨간색)
plt.plot(t_real, y_target_noisy, color='red', alpha=0.1, label=f'Raw Noisy Data ({noise_level})')

# (3) [NEW] 적응형 필터로 펴진 신호 (진한 빨간 실선)
plt.plot(t_real, y_filtered, color='red', linewidth=2, label=f'Adaptive Filtered (Win={auto_window})')

# (4) AI가 제어용으로 시뮬레이션한 모델 (파란 점선)
plt.plot(t_sim, y_sim, 'b--', linewidth=2, label=f'AI Model (Aging={ai_estimated_aging:.3f})')

# (5) 제어 명령 선 및 기존 계획 선
plt.axvline(x=55, color='gray', linestyle='--', linewidth=1, label='Legacy Plan (55min)', alpha=0.5)
plt.axvline(x=optimal_cut_point, color='blue', linestyle='--', linewidth=2, label=f'AI Control ({optimal_cut_point:.1f}min)')

# --- 피크 이동 화살표 및 하이라이트 ---
plt.annotate('', xy=(real_peak_t, real_peak_y), xytext=(new_peak_t, new_peak_y),
             arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=10))
plt.text((real_peak_t + new_peak_t)/2, (real_peak_y + new_peak_y)/2 + 0.3, 
         'Peak Shift due to Aging', ha='center', fontsize=11, fontweight='bold')

plt.text(real_peak_t - 15, real_peak_y * 0.8, 'AI SAVED:\nAdjusted for Aging', 
         color='blue', fontweight='bold', fontsize=12, ha='center')

plt.title(f"Adaptive Digital Twin: Noise σ={baseline_noise:.2f} -> Auto Window {auto_window}", fontsize=16)
plt.xlabel("Time (min)", fontsize=12)
plt.ylabel("Target Concentration", fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
# ==========================================
# 5. 시스템 성능 분석 (성적표 계산) - 가변 마진 대응판
# ==========================================

# (1) 노화도 진단 오차율 (%)
aging_error_pct = abs(real_aging - ai_estimated_aging) / real_aging * 100

# (2) 피크 시간 예측 오차 (초 단위)
true_peak_time = t_real[np.argmax(y_target_clean)]
peak_time_diff_sec = abs(true_peak_time - peak_time) * 60 

# (3) 제어 명령 정확도 계산 (가변 마진 대응)
# 정답지: 실제 노화도(real_aging)를 넣었을 때 나와야 하는 마진
true_adaptive_margin = 9.0 - (real_aging * 5.0)
true_golden_time = true_peak_time - true_adaptive_margin # 이것이 진짜 '정답' 시간입니다.

# AI의 명중 오차 (초 단위)
control_error_sec = abs(optimal_cut_point - true_golden_time) * 60

# (4) 기존 방식(55분 고정) 대비 개선 효과
legacy_error = abs(55.0 - true_golden_time)
ai_error = abs(optimal_cut_point - true_golden_time)
ai_improvement_min = legacy_error - ai_error

print("\n" + "="*50)
print("🎯 [디지털 트윈 시스템 최종 성적표 - 가변 마진 모드]")
print("-"*50)
print(f"1. 노화도 진단 정확도  : {100 - aging_error_pct:.4f} %")
print(f"   (실제: {real_aging:.2f} | AI추정: {ai_estimated_aging:.4f})")
print(f"2. 피크 시간 예측 오차 : {peak_time_diff_sec:.2f} 초")
print(f"3. 제어 명령 정확도    : AI가 실제 골든타임을 {control_error_sec:.2f}초 차이로 명중함")
print(f"4. 공정 개선 효과      : 기존 방식 대비 약 {ai_improvement_min:.1f}분 더 정확하게 수거 시작")
print("="*50)
plt.show()
