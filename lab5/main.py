import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ЗАДАНА ФУНКЦІЯ (навантаження на сервер)
def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

# ТОЧНЕ ЗНАЧЕННЯ ІНТЕГРАЛУ
a, b = 0, 24
I_exact, _ = quad(f, a, b)
print("\n=== Точне значення (еталон) ===")
print(I_exact)

# 3. СКЛАДОВА ФОРМУЛА СІМПСОНА
# n - кількість розбиттів
# =========================================================
def simpson(f, a, b, n):
    if n % 2 != 0:
        n += 1  # щоб була парність

    h = (b - a) / n  # крок
    x = np.linspace(a, b, n + 1)  # вузли
    y = f(x)

    S = y[0] + y[-1]

    # додаю внутрішні точки
    for i in range(1, n):
        if i % 2 == 0:
            S += 2 * y[i]
        else:
            S += 4 * y[i]

    return S * h / 3

# ДОСЛІДЖЕННЯ ПОХИБКИ
# будуємо графік залежності похибки від n
n_values = range(2, 100, 2)
errors = []

for n in n_values:
    I = simpson(f, a, b, n)
    error = abs(I_exact - I)
    errors.append(error)


# ПОХИБКА ДЛЯ ФІКСОВАНОГО n
n = 16
I_n = simpson(f, a, b, n)
error_n = abs(I_exact - I_n)

print("\n= Сімпсон =")
print("Інтеграл:", I_n)
print("Похибка:", error_n)


# 6. МЕТОД РУНГЕ-РОМБЕРГА
# уточнення значення інтегралу
def runge_romberg(I_h, I_h2, p):
    return I_h2 + (I_h2 - I_h) / (2**p - 1)

I_h = simpson(f, a, b, n)
I_h2 = simpson(f, a, b, 2*n)

p = 4  # порядок точності методу Сімпсона
I_rr = runge_romberg(I_h, I_h2, p)

print("\n= Рунге-Ромберг =")
print("Уточнене значення:", I_rr)
print("Похибка:", abs(I_exact - I_rr))

# 7. МЕТОД ЕЙТКЕНА
# також покращує точність і дозволяє оцінити порядок
# =========================================================
def aitken(I1, I2, I3):
    denom = (I1 - 2*I2 + I3)
    if abs(denom) < 1e-12:
        return I3
    return I3 + (I3 - I2)**2 / denom

I1 = simpson(f, a, b, n)
I2 = simpson(f, a, b, 2*n)
I3 = simpson(f, a, b, 4*n)

I_aitken = aitken(I1, I2, I3)

print("\n= Ейткен =")
print("Уточнене значення:", I_aitken)
print("Похибка:", abs(I_exact - I_aitken))

# 8. АДАПТИВНИЙ МЕТОД СІМПСОНА
# автоматично змінює крок для досягнення точності
def adaptive_simpson(f, a, b, eps):
    # локальна формула Сімпсона
    def simpson_local(f, a, b):
        c = (a + b) / 2
        return (f(a) + 4*f(c) + f(b)) * (b - a) / 6

    # рекурсивна частина
    def recursive(f, a, b, eps, whole):
        c = (a + b) / 2

        left = simpson_local(f, a, c)
        right = simpson_local(f, c, b)

        # перевірка точності
        if abs(left + right - whole) < 15 * eps:
            return left + right + (left + right - whole) / 15

        # рекурсивне ділення
        return recursive(f, a, c, eps/2, left) + \
               recursive(f, c, b, eps/2, right)

    return recursive(f, a, b, eps, simpson_local(f, a, b))

I_adaptive = adaptive_simpson(f, a, b, 1e-5)

print("\n= Адаптивний метод =")
print("Інтеграл:", I_adaptive)
print("Похибка:", abs(I_exact - I_adaptive))

print("\n=== Порівняння ===")
print("Сімпсон:", I_n)
print("Рунге-Ромберг:", I_rr)
print("Ейткен:", I_aitken)
print("Адаптивний:", I_adaptive)

# 9. ПОБУДОВА ГРАФІКА ФУНКЦІЇ
# =========================================================
x = np.linspace(0, 24, 1000)
y = f(x)

plt.figure(figsize=(6, 10))

plt.subplot(3,1,1)
plt.plot(x, y)
plt.title("Графік навантаження на сервер")
plt.xlabel("Час (год)")
plt.ylabel("f(x)")
plt.grid()

plt.subplot(3,1,2)
plt.plot(n_values, errors)
plt.xlabel("Кількість розбиттів n")
plt.ylabel("Похибка")
plt.title("Залежність похибки від n (Сімпсон)")
plt.grid()

plt.subplot(3,1,3)
plt.semilogy(n_values, errors)
plt.title("Логарифмічна залежність похибки")
plt.xlabel("n")
plt.ylabel("log(похибки)")
plt.grid()

plt.tight_layout(pad=3)
plt.show()