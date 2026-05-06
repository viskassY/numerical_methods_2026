import matplotlib.pyplot as plt
import numpy as np

# =========================
# ЦІЛЬОВІ ФУНКЦІЇ
# =========================

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def f1(x):
    return x[0]**2 + x[1] - 1

def f2(x):
    return x[0] + x[1]**2 - 1

def phi(x):
    return f1(x)**2 + f2(x)**2


# =========================
# ДОСЛІДЖУЮЧИЙ ПОШУК
# =========================
def exploratory_search(f, x_base, step, q, eps1, reduce_step=True):

    x_new = np.array(x_base, dtype=float)
    n = len(x_new)

    for i in range(n):
        success = False
        f_old = f(x_new)

        # Цикл зменшення кроку для конкретної координати i
        while step[i] >= eps1:
            # Пробний крок "+"
            x_new[i] += step[i]
            if f(x_new) < f_old:
                success = True
                break

            # Пробний крок "-"
            x_new[i] -= 2 * step[i]
            if f(x_new) < f_old:
                success = True
                break

            # Якщо обидва напрямки невдалі — повертаємося в початкову точку
            # і зменшуємо крок для цієї координати
            x_new[i] += step[i]
            step[i] = step[i] / q

        # Якщо після всіх зменшень кроку покращення не знайдено,
        # координата x_new[i] залишається рівною x_base[i]
    return x_new, step

# =========================
# МЕТОД ХУКА-ДЖИВСА
# =========================
def hooke_jeeves(f, x0, start_step, q=2.0, p=2.0, eps1=1e-6, eps2=1e-7):
    # Ініціалізація
    x_base = np.array(x0, dtype=float)
    step = np.array(start_step, dtype=float)
    trajectory = [x_base.copy()]

    k = 0
    while True:
        k += 1
        x_old_base = x_base.copy()
        f_old_base = f(x_old_base)

        # 1. Досліджуючий пошук
        x_new, updated_step = exploratory_search(f, x_old_base, step.copy(), q, eps1)

        # Перевірка чи знайшли нову точку
        if not np.array_equal(x_new, x_old_base):
            # 2. Пошук по зразку (Pattern Search)
            while True:
                x_pattern = x_new + p * (x_new - x_old_base)
                x_prev_base = x_new.copy()

                # Досліджуючий пошук з точки зразка (БЕЗ зменшення кроку)
                x_after_pattern, _ = exploratory_search(f, x_pattern, updated_step, q, eps1, reduce_step=False)

                # Додаємо кожну нову базисну точку до траєкторії
                trajectory.append(x_prev_base.copy())

                # Перевірка умови покращення
                if f(x_after_pattern) < f(x_prev_base):
                    x_old_base = x_prev_base
                    x_new = x_after_pattern
                else:
                    x_base = x_prev_base
                    break
        else:
            # Якщо покращення немає — оновлюємо кроки для наступної спроби
            x_base = x_old_base
            step = updated_step


        if np.linalg.norm(step) < eps1 and abs(f(x_base) - f_old_base) < eps2:
            break

        # Запобіжник
        if k > 1000:
            break

    with open("trajectory.txt", "w") as file:
        file.write("Step\tX1\tX2\tF(X)\n")
        for i, pt in enumerate(trajectory):
            file.write(f"{i}\t{pt[0]:.6f}\t{pt[1]:.6f}\t{f(pt):.10f}\n")

    return x_base, f(x_base), trajectory, k

# ==========================================================
# 4. ВІЗУАЛІЗАЦІЯ
# ==========================================================
def plot_results(f, trajectory, mode="system"):
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y) # сітка з координат

    if mode == "system":
        Z = f1([X, Y]) ** 2 + f2([X, Y]) ** 2 # масив значень висоти
    else:
        Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2

    plt.figure(figsize=(10, 8))
    # Малюємо рельєф, створення ліній рівня
    cp = plt.contour(X, Y, Z, levels=np.logspace(-2, 3, 20), cmap='viridis', alpha=0.5)
    plt.clabel(cp, inline=1, fontsize=8)

    if mode == "system":
        # f1(x,y) = x^2 + y - 1 = 0
        plt.contour(X, Y, X ** 2 + Y - 1, levels=[0], colors='red', linestyles='dashed')
        # f2(x,y) = x + y^2 - 1 = 0
        plt.contour(X, Y, X + Y ** 2 - 1, levels=[0], colors='blue', linestyles='dashed')
        plt.title("Метод Хука-Дживса: Розв'язок системи рівнянь (Перетин f1 та f2)")
    else:
        plt.title("Метод Хука-Дживса: Мінімізація функції Розенброка")

    # Малюємо шлях пошуку
    traj = np.array(trajectory)
    plt.plot(traj[:, 0], traj[:, 1], 'mo-', markersize=4, label='Траєкторія (кроки)')
    plt.plot(traj[0, 0], traj[0, 1], 'go', markersize=8, label='Старт')
    plt.plot(traj[-1, 0], traj[-1, 1], 'rx', markersize=10, label='Знайдений мінімум')

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.show()



# мейн

if __name__ == "__main__":
    mode = "system" #system

    if mode == "system":
        target_f = phi
        x0 = [-1.0, -1.0]
        step = [0.05, 0.05]
    else:
        target_f = rosenbrock
        x0 = [-1.2, 0.0]
        step = [0.1, 0.1]

    # Запуск алгоритму
    xmin, fmin, trajectory, total_steps = hooke_jeeves(
        target_f, x0, step, q=2.0, p=2.0, eps1=1e-6, eps2=1e-7
    )

    print(f"--- РЕЗУЛЬТАТИ ({mode}) ---")
    print(f"Точка мінімуму: {xmin}")
    print(f"Значення функції: {fmin:.10f}")
    print(f"Кількість ітерацій: {total_steps}")

    # Відображення
    plot_results(target_f, trajectory, mode=mode)