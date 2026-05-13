import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# ДИФЕРЕНЦІАЛЬНЕ РІВНЯННЯ
# y' = x + y
# y(0) = 1
# =========================================================

def f(x, y):
    return x + y

def exact(x):
    return -x - 1 + 2 * np.exp(x)

a = 0.0
b = 1.0
y0 = 1.0          # початкова умова
h = 0.1           # крок
eps = 1e-5        # точність для adaptive step



# АНАЛІТИЧНИЙ РОЗВ'ЯЗОК ДЛЯ ГРАФІКІВ
x_exact = np.linspace(a, b, 200)
y_exact = exact(x_exact)

# ОДИН КРОК RK4
# Використовується:
# - для старту Adams
# - у повному методі RK4
# =========================================================

def rk4_step(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + h / 2,y + h * k1 / 2)
    k3 = f(x + h / 2,y + h * k2 / 2)
    k4 = f(x + h,y + h * k3)
    return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

# СТАРТОВА ТОЧКА ДЛЯ ADAMS
x_vals = [a]
y_vals = [y0]
x_vals.append(a + h)
y_start = rk4_step(f, a, y0, h)
y_vals.append(y_start)

# МЕТОД ADAMS 2 Predictor-Corrector
def adams2(f, a, b, y0, h):

    N = int((b - a) / h)
    xs = x_vals.copy()
    ys = y_vals.copy()
    ys_pr = [ys[1]]
    for n in range(1, N):
        x_next = xs[n] + h
        # похідні
        f_n = f(xs[n], ys[n])
        f_nm1 = f(xs[n - 1], ys[n - 1])
        # predictor
        y_next_pr = ys[n] + h / 2 * (
            3 * f_n - f_nm1
        )
        # corrector
        f_pr = f(x_next, y_next_pr)
        y_next = ys[n] + h / 2 * (
            f_pr + f_n
        )

        xs.append(x_next)
        ys.append(y_next)
        ys_pr.append(y_next_pr)

    return (
        np.array(xs),
        np.array(ys),
        np.array(ys_pr)
    )


# РОЗВ'ЯЗОК ADAMS
xs, ys, ys_pr = adams2(f, a, b, y0, h)


# ПОХИБКИ ADAMS
# глобальна похибка
phi_exact = ys - exact(xs)

# predictor-corrector estimate
phi_est = np.zeros_like(ys)
phi_est[1:] = ys[1:] - ys_pr

# ADAPTIVE STEP FOR ADAMS
def adams2_autostep(f, a, b, y0, eps):
    h_cur = 0.1
    # стартові точки
    x_prev = a
    y_prev = y0

    x_cur = x_prev + h_cur
    y_cur = rk4_step(f, x_prev, y_prev, h_cur)

    xs_auto = [x_prev, x_cur]
    ys_auto = [y_prev, y_cur]
    hs_auto = [h_cur, h_cur]

    while x_cur < b - 1e-12:
        # похідні
        f_n = f(
            xs_auto[-1],
            ys_auto[-1]
        )
        f_nm1 = f(
            xs_auto[-2],
            ys_auto[-2]
        )
        # predictor
        y_pr = ys_auto[-1] + h_cur / 2 * (
            3 * f_n - f_nm1
        )
        # corrector
        f_pr = f(
            xs_auto[-1] + h_cur,
            y_pr
        )

        y_cor = ys_auto[-1] + h_cur / 2 * (
            f_pr + f_n
        )

        # оцінка похибки
        err = abs(y_cor - y_pr) / 3

        # якщо похибка завелика
        if err > eps:
            h_cur /= 2
            continue

        # приймаємо точку
        x_cur = xs_auto[-1] + h_cur
        xs_auto.append(x_cur)
        ys_auto.append(y_cor)
        hs_auto.append(h_cur)

        # якщо похибка мала
        if err < eps / 10:
            h_cur *= 2

    return (
        np.array(xs_auto),
        np.array(ys_auto),
        np.array(hs_auto)
    )

# ADAMS WITH AUTO STEP
xs_a, ys_a, hs_a = adams2_autostep(f, a, b, y0, eps)

# RUNGE-KUTTA 4
def runge_kutta4(f, a, b, y0,h):

    N = int((b - a) / h)
    xs = np.zeros(N + 1)
    ys = np.zeros(N + 1)

    xs[0] = a
    ys[0] = y0

    for n in range(N):
        x = xs[n]
        y = ys[n]

        ys[n + 1] = rk4_step(f,x,y,h)
        xs[n + 1] = x + h
    return xs, ys


# RK4 РОЗВ'ЯЗОК
xs_rk, ys_rk = runge_kutta4(f,a,b,y0,0.01)

# ПОХИБКА RK4
phi_rk = ys_rk - exact(xs_rk)
# ПОХИБКА ПО РУНГЕ

def local_error_runge(f,a,b,y0,h):

    xs_h, ys_h = runge_kutta4(f,a,b,y0,h)
    xs_h2, ys_h2 = runge_kutta4(f,a,b,y0,h / 2)
    ys_h2_even = ys_h2[::2]

    phi_runge = (
        16 / 15
    ) * (
        ys_h2_even - ys_h
    )
    return xs_h, phi_runge

xs_r, phi_r = local_error_runge(f,a,b,y0,0.01)

# ADAPTIVE STEP FOR RK4
def rk4_autostep(f,a,b,y0,eps):

    h_cur = 0.01
    x_cur = a
    y_cur = y0

    xs = [x_cur]
    ys = [y_cur]
    hs = [h_cur]

    while x_cur < b - 1e-12:
        # розв'язок з h/2
        xs_half, ys_half = runge_kutta4(f,x_cur,x_cur + h_cur,y_cur,h_cur / 2)
        y_half = ys_half[-1]

        # розв'язок з h
        xs_full, ys_full = runge_kutta4(f,x_cur,x_cur + h_cur,y_cur,h_cur)
        y_full = ys_full[-1]

        # оцінка похибки Рунге
        err = abs(y_half - y_full) / 15

        # якщо похибка завелика
        if err > eps:
            h_cur /= 2
            continue

        # приймаємо точку
        x_cur += h_cur
        y_cur = y_half

        xs.append(x_cur)
        ys.append(y_cur)
        hs.append(h_cur)

        # якщо похибка мала
        if err < eps / 10:
            h_cur *= 2

    return (
        np.array(xs),
        np.array(ys),
        np.array(hs)
    )

# RK4 WITH AUTO STEP
xs_auto, ys_auto, hs_auto = rk4_autostep(f,a,b,y0,1e-4)

# ГРАФІКИ — ЧАСТИНА 1 (ADAMS)\
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.plot( x_exact,y_exact,label='Точний')

plt.plot( xs,ys,label='Адамс 2')
plt.legend()
plt.grid(True)
plt.title("Ч.1: Розв'язки (Адамс 2)")

# ПОХИБКИ
plt.subplot(2, 2, 2)

plt.plot(xs,np.abs(phi_exact),label='φ exact')
plt.plot(xs,np.abs(phi_est),label='φ estimate')
plt.yscale('log')
plt.legend()
plt.grid(True)

plt.title("Ч.1: Похибки")

# ADAPTIVE STEP
plt.subplot(2, 2, 3)
plt.plot(x_exact,y_exact,alpha=0.5,label='Точний')
plt.plot(xs_a,ys_a,label='Адамс (auto-h)')
plt.legend()
plt.grid(True)
plt.title("Ч.1: Автовибір кроку")


# h(x)
plt.subplot(2, 2, 4)

plt.step( xs_a,hs_a,where='post')
plt.grid(True)
plt.title("Ч.1: h(x)")
plt.xlabel("x")
plt.ylabel("h")
plt.tight_layout()
plt.show()

# ГРАФІКИ — ЧАСТИНА 2 (RK4)
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.plot(x_exact,y_exact,label='Точний')
plt.plot( xs_rk,ys_rk,label='RK4')
plt.legend()
plt.grid(True)
plt.title("Ч.2: Розв'язки (RK4)")


# GLOBAL ERROR
plt.subplot(2, 2, 2)

plt.plot( xs_rk,np.abs(phi_rk))
plt.yscale('log')
plt.grid(True)
plt.title("Ч.2: Global error")


# RUNGE ERROR
plt.subplot(2, 2, 3)

plt.plot(xs_r,np.abs(phi_r))
plt.yscale('log')
plt.grid(True)
plt.title("Ч.2: Runge error")

# h(x)
plt.subplot(2, 2, 4)

plt.step(xs_auto,hs_auto,where='post')
plt.grid(True)
plt.title("Ч.2: h(x)")
plt.xlabel("x")
plt.ylabel("h")
plt.tight_layout()
plt.show()

# ТАБЛИЦЯ RK4
print(f"{'x':>6} {'y_exact':>12} {'y_rk4':>12} {'φ_exact':>12}")

for i in range(len(xs_rk)):
    print(
        f"{xs_rk[i]:6.2f} "
        f"{exact(xs_rk[i]):12.6f} "
        f"{ys_rk[i]:12.6f} "
        f"{phi_rk[i]:12.2e}"
    )