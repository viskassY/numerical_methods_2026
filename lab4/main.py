import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Функція
# -----------------------------
def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def dM_exact(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

t0 = 1
exact = dM_exact(t0) # точне значення у похідній при t = 1

# -----------------------------
# Центральна різниця
# -----------------------------
def derivative_central(t, h):
    return (M(t + h) - M(t - h)) / (2 * h)

# -----------------------------
# Дослідження похибки
# -----------------------------
h_values = [10**(-k) for k in range(1, 12)]
errors = []
errors_runge = []
errors_aitken = []

for h in h_values:
    D_h = derivative_central(t0, h)
    D_2h = derivative_central(t0, 2*h)
    D_4h = derivative_central(t0, 4*h)

    # базова похибка
    errors.append(abs(D_h - exact))

    # Рунге
    D_rr = D_h + (D_h - D_2h) / 3
    errors_runge.append(abs(D_rr - exact))

    # Ейткен
    D_aitken = (D_2h**2 - D_4h * D_h) / (2*D_2h - (D_4h + D_h))
    errors_aitken.append(abs(D_aitken - exact))

# Фіксований крок
h = 0.001

D_h = derivative_central(t0, h)
#D_h2 = derivative_central(t0, h/2)
D_2h = derivative_central(t0, 2*h)
D_4h = derivative_central(t0, 4*h)


# Рунге–Ромберг
p = 2 # порядок точності
#D_rr = D_h2 + (D_h2 - D_h) / (2**p - 1)
D_rr = D_h + (D_h - D_2h) / 3

# Ейткен
D_aitken = (D_2h**2 - D_4h * D_h) / (2*D_2h - (D_4h + D_h))

p_est = (1/np.log(2)) * np.log(abs((D_4h - D_2h)/(D_2h - D_h))) # розрахунковий порядок точності

# -----------------------------
# Дані для графіків
# -----------------------------
t = np.linspace(0, 20, 200)

plt.figure(figsize=(10, 10))

# 1. Похибка
plt.subplot(2, 1, 1)
plt.loglog(h_values, errors, marker='o', label="Центральна різниця")
plt.loglog(h_values, errors_runge, marker='o', label="Рунге-Ромберг")
plt.loglog(h_values, errors_aitken, marker='o', label="Ейткен")

plt.title("Похибка vs крок")
plt.xlabel("Крок диференціювання h")
plt.ylabel("error")
plt.legend()
plt.grid()

# 2. Функція
plt.subplot(2, 1, 2)
plt.plot(t, M(t), label="M(t) — Вологість ґрунту (%)")
plt.plot(t, dM_exact(t), linestyle='--', label="M'(t) — Швидкість зміни вологості")
plt.scatter(t0, exact, label="t = 1")
plt.title("Модель вологості ґрунту та швидкість її зміни")
plt.xlabel("Час t (умовні одиниці)")
plt.ylabel("Значення M(t) та M'(t)")
plt.legend()
plt.grid()
plt.tight_layout(pad=5)
plt.show()

# -----------------------------
# Вивід результатів
# -----------------------------
print("Порядок точності (за Ейткеном) p = ", p_est)
print("y'(h) =", D_h)
print("y'(2h) =", D_2h)
print("y'(4h) =", D_4h)

print("Точне значення:", exact)
print("Runge-Romberg:", D_rr)
print("Aitken:", D_aitken)

print("Похибка Aitken:", abs(D_aitken - exact))
print("Похибка Runge:", abs(D_rr - exact))