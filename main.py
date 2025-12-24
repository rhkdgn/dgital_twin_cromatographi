import matplotlib.pyplot as plt
from model import VirtualColumn

print("--- 💥 연구 Step 1: 노화(Aging)에 따른 공정 실패 시뮬레이션 💥 ---")

# ==========================================
# 1. 기준 실험 (Reference): 새 컬럼 (Aging = 1.0)
# ==========================================
# 우리가 "평소에 기대하는" 정상적인 상황
col_new = VirtualColumn(N=50, V_total=10.0, Q=1.0)
col_new.aging_factor = 1.0 
time_new, y_target_new, y_imp_new = col_new.run_simulation(t_max=100)

# ==========================================
# 2. 사고 실험 (Disaster): 늙은 컬럼 (Aging = 0.65)
# ==========================================
# 연구원이 모르는 사이에 망가진 상황 -> 피크가 앞으로 당겨짐
col_old = VirtualColumn(N=50, V_total=10.0, Q=1.0)
col_old.aging_factor = 0.65 
time_old, y_target_old, y_imp_old = col_old.run_simulation(t_max=100)

# ==========================================
# 3. 결과 비교 그래프 (논문용 그림 1)
# ==========================================
plt.figure(figsize=(12, 6))

# (1) 정상 상태 (점선으로 표시 - 기준점)
plt.plot(time_new, y_target_new, 'b:', label='Expected (New Column)', linewidth=1.5, alpha=0.6)

# (2) 노화 상태 (실선으로 표시 - 실제 상황)
plt.plot(time_old, y_target_old, 'r-', label='Real (Old Column)', linewidth=2.0)

# (3) 연구원의 기존 수거 타이밍 (예: 55분부터 받으려고 계획함)
plt.axvline(x=55, color='green', linestyle='--', label='Cut-point (Plan)')
plt.text(56, 2.0, 'Legacy Cut-point', color='green')

# 그래프 꾸미기
plt.title('Why do we need AI? (Effect of Aging)', fontsize=15)
plt.xlabel('Time (min)')
plt.ylabel('Concentration')
plt.legend()
plt.grid(True)

# 화살표로 "이동했다"는 걸 강조 (시각적 효과)
plt.annotate('Shifted!', xy=(42, 1.5), xytext=(55, 1.8),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, color='red')

print("--- 그래프 출력 ---")
plt.show()