import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1. ТРАНСЦЕНДЕНТНА ФУНКЦІЯ
# =========================================================
def f(x):
    return x**3 - x - 2

def df(x):
    return 3*x**2 - 1

def d2f(x):
    return 6*x


# =========================================================
# 2. ТАБУЛЯЦІЯ + ВИДІЛЕННЯ КОРЕНІВ
# =========================================================
def tabulate(a, b, h):
    x_vals = np.arange(a, b + h, h)

    with open("table.txt", "w") as file:
        for x in x_vals:
            file.write(f"{x:.5f} {f(x):.5f}\n")
    # пошук абсцис точок перетину, перевіряє зміну знаку
    intervals = []
    for i in range(len(x_vals)-1):
        if f(x_vals[i]) * f(x_vals[i+1]) < 0:
            intervals.append((x_vals[i], x_vals[i+1]))

    return intervals


# =========================================================
# 3. МЕТОДИ РОЗВ’ЯЗКУ
# =========================================================

# 3.1 Проста ітерація
def simple_iteration(x0, tau, eps=1e-10, max_iter=100):
    x = x0
    for i in range(max_iter):
        x_new = x + tau * f(x)
         # критерій зупинки
        if abs(x_new - x) < eps and abs(f(x_new)) < eps:
            return x_new, i+1

        x = x_new
    return x, max_iter


# 3.2 Ньютон
def newton(x0, eps=1e-10, max_iter=100):
    x = x0
    for i in range(max_iter):
        x_new = x - f(x)/df(x)

        if abs(x_new - x) < eps and abs(f(x_new)) < eps:
            return x_new, i+1

        x = x_new
    return x, max_iter


# 3.3 Чебишев
def chebyshev(x0, eps=1e-10, max_iter=100):
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        d2fx = d2f(x)

        x_new = x - fx/dfx - (d2fx * fx**2)/(2 * dfx**3)

        if abs(x_new - x) < eps and abs(f(x_new)) < eps:
            return x_new, i+1

        x = x_new
    return x, max_iter


# 3.4 Хорди
def chord(x0, x1, eps=1e-10, max_iter=100):
    for i in range(max_iter):
        denominator = f(x1) - f(x0)
        if abs(denominator) < 1e-15:  # Якщо значення функції майже однакові
            return x1, i

        x_new = x1 - f(x1)*(x1 - x0)/(f(x1) - f(x0))

        if abs(x_new - x1) < eps and abs(f(x_new)) < eps:
            return x_new, i+1

        x0, x1 = x1, x_new
    return x1, max_iter


# 3.5 Параболи
def parabol(x0, x1, x2, eps=1e-10, max_iter=100):
    for i in range(max_iter):
        f0, f1, f2 = f(x0), f(x1), f(x2)

        # Перевірка на однакові x, щоб не було ділення на нуль при обчисленні a1, a2
        if abs(x1 - x0) < 1e-15 or abs(x2 - x1) < 1e-15 or abs(x2 - x0) < 1e-15:
            return x2, i

        a1 = (f1 - f0)/(x1 - x0) # нахил
        a2 = ((f2 - f0)/(x2 - x0) - a1)/(x2 - x1) # вигин

        D = a1**2 - 4*f2*a2
        if D < 0:
            return None, i

        # Перевірка знаменника перед обчисленням x_new
        denominator = a1 + np.sign(a1) * np.sqrt(D)
        if abs(denominator) < 1e-15:
            return x2, i

        x_new = x2 - (2*f2)/(a1 + np.sign(a1)*np.sqrt(D))

        if abs(x_new - x2) < eps and abs(f(x_new)) < eps:
            return x_new, i+1

        x0, x1, x2 = x1, x2, x_new
    return x2, max_iter


# 3.6 Зворотна інтерполяція
def inverse_interpolation(x0, x1, x2, eps=1e-10, max_iter=100):
    for i in range(max_iter):
        f0, f1, f2 = f(x0), f(x1), f(x2)

        # Перевірка, щоб знаменники не були нульовими
        d01, d02, d12 = f0 - f1, f0 - f2, f1 - f2
        if abs(d01) < 1e-15 or abs(d02) < 1e-15 or abs(d12) < 1e-15:
            return x2, i

        x_new = (x0*f1*f2/((f0-f1)*(f0-f2)) +
                 x1*f0*f2/((f1-f0)*(f1-f2)) +
                 x2*f0*f1/((f2-f0)*(f2-f1)))

        if abs(x_new - x2) < eps and abs(f(x_new)) < eps:
            return x_new, i+1

        x0, x1, x2 = x1, x2, x_new
    return x2, max_iter


# =========================================================
# 4. АЛГЕБРАЇЧНЕ РІВНЯННЯ (ГОРНЕР)
# =========================================================
# функція для обчислення многочлена (7 пункт)
def horner(coeffs, x):
    result = coeffs[0]
    for c in coeffs[1:]:
        result = result*x + c
    return result

def derivative_coeffs(coeffs):
    n = len(coeffs)
    return [coeffs[i]*(n-i-1) for i in range(n-1)] #коефіцієнт на степінь

def read_coeffs(filename):
    with open(filename) as f:
        return list(map(float, f.read().split()))

def newton_horner(coeffs, x0, eps=1e-10):
    x = x0
    for i in range(100):
        fx = horner(coeffs, x)
        dfx = horner(derivative_coeffs(coeffs), x)

        x_new = x - fx/dfx

        if abs(x_new - x) < eps:
            return x_new, i+1

        x = x_new
    return x, 100


# =========================================================
# 5. МЕТОД ЛІНА
# =========================================================

def lin_method(coeffs, alpha, beta, eps=1e-5, max_iter=200):
    n = len(coeffs) - 1

    for k in range(max_iter):
        b = [0]*(n+1)
        c = [0]*(n+1)

        b[0] = coeffs[0]
        b[1] = coeffs[1] + alpha*b[0]

        for i in range(2, n+1):
            b[i] = coeffs[i] + alpha*b[i-1] + beta*b[i-2]

        c[0] = b[0]
        c[1] = b[1] + alpha*c[0]

        for i in range(2, n):
            c[i] = b[i] + alpha*c[i-1] + beta*c[i-2]

        D = c[n-2]**2 - c[n-3]*c[n-1]

        if abs(D) < 1e-12:
            continue

        dalpha = (-b[n-1]*c[n-2] + b[n]*c[n-3]) / D
        dbeta  = (-b[n]*c[n-2] + b[n-1]*c[n-1]) / D

        alpha += dalpha
        beta += dbeta

        if abs(dalpha) < eps and abs(dbeta) < eps:
            break

    D = alpha**2 - 4*beta

    if D >= 0:
        return None

    real = -alpha/2
    imag = np.sqrt(abs(D))/2

    return (real + 1j*imag, real - 1j*imag), k+1


# =========================================================
# 6. ГРАФІК АЛГЕБРАЇЧНОГО РІВНЯННЯ
# =========================================================

def plot_poly():
    def poly(x):
        return x**3 - 2*x**2 + 2*x - 2

    x = np.linspace(-2, 3, 200)
    y = poly(x)

    plt.figure()
    plt.plot(x, y)
    plt.title("Графік алгебраїчного рівняння")
    plt.grid()
    plt.show()


# =========================================================
# 7. MAIN
# =========================================================

# табуляція
intervals = tabulate(-3, 3, 0.1)
print("Інтервали коренів:", intervals)

x0, x1 = intervals[0]
roots = []

for a, b in intervals:
    mid = (a + b) / 2

    if df(mid) > 0:
        print(f"Корінь (зростає): {mid:.4f}")
        roots.append(mid)
    else:
        print(f"Корінь (спадає): {mid:.4f}")
        roots.append(mid)

# беремо 2 різних
x0 = roots[0]
x1 = roots[-1]

tau = -0.1

print("\n=== МЕТОДИ ===")

# 1. Беремо знайдений центр інтервалу як базову точку
base_x = roots[0]

# 2. Викликаємо методи з "розкидом" точок
r, it = simple_iteration(base_x, tau)
print(f"Проста ітерація: x = {r:.6f}, ітерацій = {it}")

r, it = newton(base_x)
print(f"Ньютон: x = {r:.6f}, ітерацій = {it}")

r, it = chebyshev(base_x)
print(f"Чебишев: x = {r:.6f}, ітерацій = {it}")

r, it = chord(base_x - 0.1, base_x + 0.1)
print(f"Хорди: x = {r:.6f}, ітерацій = {it}")

res = parabol(base_x - 0.1, base_x, base_x + 0.1)
if res[0] is None:
    print("Параболи: не застосовується (D < 0)")
else:
    print(f"Параболи: x = {res[0]:.6f}, ітерацій = {res[1]}")

r, it = inverse_interpolation(base_x - 0.1, base_x, base_x + 0.1)
print(f"Зворотна інтерполяція: x = {r:.6f}, ітерацій = {it}")
# алгебраїчне рівняння 3-го порядку
coeffs = read_coeffs("coeffs.txt")

real_root, it = newton_horner(coeffs, 1)
print("\nДійсний корінь:", real_root, "ітерацій:", it)

print("\n=== МЕТОД ЛІНА ===")

initial_guesses = [
    (0.5, 0.5),
    (0, 1),
    (-1, 1),
    (1, 1)
]

for alpha0, beta0 in initial_guesses:
    result = lin_method(coeffs, alpha0, beta0)

    if result is not None:
        complex_roots, it_lin = result

        r1, r2 = complex_roots

        print(f"Зійшлось при α={alpha0}, β={beta0}")
        print(f"Корені: {r1.real:.4f} + {r1.imag:.4f}i, {r2.real:.4f} - {r2.imag:.4f}i")
        print(f"Ітерацій: {it_lin}")
        break
else:
    print("Метод Ліна не зійшовся")

# графік
plot_poly()