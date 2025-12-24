import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from model import VirtualColumn

# ==========================================
# 1. [문제 출제] 현실 세계(Real Plant) 데이터 만들기
# ==========================================
print("--- 🕵️‍♂️ Step 2: AI 탐정 수사 시작 ---")
print("1. 현실 데이터 생성 중... (정답은 우리가 몰래 설정한 0.65)")

# 진짜 공장 (우리가 맞혀야 할 대상)
real_plant = VirtualColumn(N=50, V_total=10.0, Q=1.0)
real_plant.aging_factor = 0.65  # <--- [정답] AI는 이 숫자를 모름!
t_real, y_target_real, _ = real_plant.run_simulation(t_max=100) # 불순물은 무시(_)

# ==========================================
# 2. [두뇌] 오차 계산 함수 (Loss Function)
# ==========================================
def calculate_error(guess):
    """
    AI가 "노화가 이정도 아닐까?"라고 추측한 값(guess)을 시뮬레이션해보고,
    현실 데이터랑 얼마나 틀렸는지 점수(오차)를 매기는 함수
    """
    current_aging_guess = guess[0]
    
    if current_aging_guess <= 0.0 or current_aging_guess > 2.0:
        return 999999999.0
    
    # AI의 상상(Simulation)
    sim_model = VirtualColumn(N=50, V_total=10.0, Q=1.0)
    sim_model.aging_factor = current_aging_guess # 추측값 적용
    _, y_target_sim, _ = sim_model.run_simulation(t_max=100)
    
    # 현실 vs 상상 차이 계산 (MSE: Mean Squared Error)
    # 그래프가 겹칠수록 0에 가까워짐
    error = np.mean((y_target_real - y_target_sim)**2)
    return error

# ==========================================
# 3. [최적화] 범인 찾기 (Smart Start 전략)
# ==========================================
print("2. AI가 범인을 찾는 중... (자동 위치 탐색 & 정밀 타격)")

# --- [전략] 1. 대충 훑어보기 (Coarse Search) ---
# 사람이 찍지 않고, 컴퓨터가 0.1 ~ 2.0 사이를 듬성듬성 찔러봅니다.
# 나중에 이 부분이 '딥러닝 예측'으로 대체될 명당 자리입니다!
search_grid =np.linspace(0.1, 2.0, 20)
best_guess = 1.0          # 일단 1.0이라고 가정
min_error = 999999999.0   # 에러 초기값 (무한대)

print(f"   >> 탐색 후보: {search_grid}")

for g in search_grid:
    # 각 후보 지점에서 에러가 얼마나 큰지 맛만 봅니다.
    # 주의: calculate_error는 리스트 형태([g])를 원하므로 대괄호 필수!
    current_error = calculate_error([g])
    
    # "어? 여기가 에러가 더 작네?" 싶으면 그곳을 출발점으로 찜합니다.
    if current_error < min_error:
        min_error = current_error
        best_guess = g

print(f"   >> 가장 유력한 출발점 발견: {best_guess} (에러: {min_error:.4f})")
print(f"   >> 여기서부터 정밀 탐색(minimize) 시작합니다!")

# --- [전략] 2. 정밀 타격 (Fine Tuning) ---
# 찾은 명당 자리(best_guess)에서 출발하니까 길을 잃을 일이 없습니다.
# method='Nelder-Mead' 그대로 사용!
result = minimize(calculate_error, [best_guess], method='Nelder-Mead', tol=1e-5)

estimated_aging = result.x[0]

# ==========================================
# 4. 결과 발표 및 검증
# ==========================================
print("-" * 30)
print(f"✅ 수사 종료!")
print(f"🕵️ AI 추정값 : {estimated_aging:.5f}")
print(f"🗝️ 실제 정답 : 0.65000")
print(f"📉 오차(Error): {abs(estimated_aging - 0.65):.5f}")
print("-" * 30)

# 검증 그래프 그리기
# AI가 찾은 값으로 최종 시뮬레이션 돌려서 겹쳐보기
best_model = VirtualColumn(N=50, V_total=10.0, Q=1.0)
best_model.aging_factor = estimated_aging
_, y_target_est, _ = best_model.run_simulation(t_max=100)

plt.figure(figsize=(10, 6))
# 현실 (파란 점선)
plt.plot(t_real, y_target_real, 'b:', linewidth=4, label='Real Data (Sensor)', alpha=0.5)
# AI 추정 (빨간 실선)
plt.plot(t_real, y_target_est, 'r-', linewidth=2, label=f'AI Estimation (Aging={estimated_aging:.2f})')

plt.title("Step 2 Result: Digital Twin Diagnostics")
plt.xlabel("Time (min)")
plt.ylabel("Target Concentration")
plt.legend()
plt.grid(True)
plt.show()