import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from model import VirtualColumn

print("---AI 자율 제어 시스템 가동---")

# ==========================================
# 1. 상황 설정: 아무도 모르게 망가진 공장 (현실)
# ==========================================
real_aging = 0.65  
real_plant = VirtualColumn(N=50, V_total=10.0, Q=1.0)
real_plant.aging_factor = real_aging

# 현실 센서 데이터 생성 (실제 공장 상황)
t_real, y_target_real, _ = real_plant.run_simulation(t_max=100)

print(f"1. 상황 발생: 컬럼 노화 진행됨 (실제 Aging Factor={real_aging})")

# ==========================================
# 2. AI 진단 (Smart Start + 정밀 탐색)
# ==========================================
def calculate_error(guess):
    current_guess = guess[0]
    # [안전장치] 탐색 범위 제한
    if current_guess <= 0.0 or current_guess > 2.0:
        return 999999999.0
        
    sim = VirtualColumn(N=50, V_total=10.0, Q=1.0)
    sim.aging_factor = current_guess
    _, y_sim, _ = sim.run_simulation(t_max=100)
    return np.mean((y_target_real - y_sim)**2)

# --- [전략 1] 촘촘하게 훑기 (Coarse Search) ---
print("   >> AI가 전역 탐색을 통해 최적의 출발점을 찾는 중...")
search_grid = np.linspace(0.1, 2.0, 20)
best_guess = 1.0
min_err = 999999999.0

for g in search_grid:
    err = calculate_error([g])
    if err < min_err:
        min_err = err
        best_guess = g

# --- [전략 2] 명당에서 정밀 탐색 (Fine Tuning) ---
res = minimize(calculate_error, [best_guess], method='Nelder-Mead', tol=1e-5)
ai_estimated_aging = res.x[0]

print(f"2. AI 진단 완료: {best_guess}에서 시작하여 최종 {ai_estimated_aging:.4f}를 찾아냄")

# ==========================================
# 3. 골든타임(Peak Time) 예측 및 제어 명령
# ==========================================
# 진단된 결과로 미래 시뮬레이션
ai_model = VirtualColumn(N=50, V_total=10.0, Q=1.0)
ai_model.aging_factor = ai_estimated_aging
t_sim, y_sim, _ = ai_model.run_simulation(t_max=100)

# 피크 시간(가장 많이 쏟아지는 지점) 찾기
peak_index = np.argmax(y_sim)
peak_time = t_sim[peak_index]

# 제어 전략: 피크 5분 전부터 밸브 열기 (Golden Time)
optimal_cut_point = peak_time - 5.0 

print(f"3. 제어 명령: 기존 55분 계획 폐기 -> '{optimal_cut_point:.1f}분'에 수거 시작!")

# ==========================================
# 4. 결과 시각화 (최종본: 피크 정점 연결형)
# ==========================================
# [기준점] 새 제품(Aging=1.0) 데이터 생성
new_plant = VirtualColumn(N=50, V_total=10.0, Q=1.0)
new_plant.aging_factor = 1.0
t_new, y_new, _ = new_plant.run_simulation(t_max=100)

# 각 그래프의 피크(최고점) 좌표 찾기
new_peak_idx = np.argmax(y_new)
new_peak_t, new_peak_y = t_new[new_peak_idx], y_new[new_peak_idx]

real_peak_idx = np.argmax(y_target_real)
real_peak_t, real_peak_y = t_real[real_peak_idx], y_target_real[real_peak_idx]

plt.figure(figsize=(14, 7))

# (1) 새 제품 상태 (연한 회색 실선)
plt.plot(t_new, y_new, color='gray', linewidth=1.5, label='Brand New Column (Aging=1.0)', alpha=0.4)

# (2) 현재 노후화된 현실 (빨간 실선)
plt.plot(t_real, y_target_real, 'r-', linewidth=3, label=f'Real Production (Aging={real_aging})')

# (3) 과거 기준선 (55분)
plt.axvline(x=55, color='gray', linestyle='--', linewidth=1.5, label='Legacy Plan (55min)', alpha=0.7)

# (4) AI 자율 제어 선
plt.axvline(x=optimal_cut_point, color='blue', linestyle='--', linewidth=2, label=f'AI Control ({optimal_cut_point:.1f}min)')

# --- [하이라이트] 피크 정점끼리 화살표 연결 ---
plt.annotate('', 
             xy=(real_peak_t, real_peak_y),      # 화살표 끝 (현재 피크)
             xytext=(new_peak_t, new_peak_y),    # 화살표 시작 (새 피크)
             arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=10))

plt.text((real_peak_t + new_peak_t)/2, (real_peak_y + new_peak_y)/2 + 0.2, 
         'Peak Shifted due to Aging', ha='center', fontsize=11, fontweight='bold')

# --- 텍스트 및 레이아웃 설정 ---
plt.text(real_peak_t - 15, real_peak_y * 0.8, 'AI SAVED:\nAdjusted for Aging', 
         color='blue', fontweight='bold', fontsize=12, ha='center')

plt.title("Digital Twin Factory: Real-time Aging Diagnosis & Control", fontsize=16, pad=20)
plt.xlabel("Time (min)", fontsize=12)
plt.ylabel("Target Concentration", fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()