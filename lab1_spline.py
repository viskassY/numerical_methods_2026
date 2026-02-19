import numpy as np
import matplotlib.pyplot as plt
import requests

url = (
    "https://api.open-elevation.com/api/v1/lookup?"
    "locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|"
    "48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|"
    "48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|"
    "48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|"
    "48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|"
    "48.160580,24.500537|48.160250,24.500106"
)

response = requests.get(url)
results = response.json()["results"]

coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = np.array([p["elevation"] for p in results])
n = len(results)

# 2. Кумулятивна відстань

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

distances = [0]
for i in range(1, n):
    distances.append(distances[-1] + haversine(*coords[i-1], *coords[i]))
distances = np.array(distances)

print("\nКумулятивна відстань (м):")
for i, d in enumerate(distances):
    print(f"{i}: {d:.2f}")

# 3. Підбір вузлів для сплайнів

def take_nodes_by_distance(distances, elevations, k):
    x_new = np.linspace(distances[0], distances[-1], k)
    y_new = np.interp(x_new, distances, elevations)
    return x_new, y_new

x10, y10 = take_nodes_by_distance(distances, elevations, 10)
x15, y15 = take_nodes_by_distance(distances, elevations, 15)
x20, y20 = take_nodes_by_distance(distances, elevations, 20)


# 4. Запис табуляції в файл

with open("/Users/Admin/Desktop/labku/2COURSE/2TERM/Metods/tabulation.txt", "w", encoding="utf-8") as f:
    f.write("№\tLatitude\tLongitude\tElevation(m)\tDistance(m)\n")
    for i in range(n):
        f.write(f"{i}\t{coords[i][0]:.6f}\t{coords[i][1]:.6f}\t"
                f"{elevations[i]:.2f}\t{distances[i]:.2f}\n")

# 5. Метод прогонки 

def progonka(y, h):
    N = len(y) - 1
    a = y[:-1].copy()
    b = np.zeros(N)
    c = np.zeros(N+1)
    d = np.zeros(N)

    alpha = np.zeros(N-1)
    beta = np.zeros(N-1)
    gamma = np.zeros(N-1)
    delta = np.zeros(N-1)

    for i in range(1, N):
        alpha[i-1] = h[i-1]
        beta[i-1] = 2 * (h[i-1] + h[i])
        gamma[i-1] = h[i]
        delta[i-1] = 3 * ((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])

    # Пряма прогонка
    P = np.zeros(N-1)
    Q = np.zeros(N-1)
    P[0] = -gamma[0]/beta[0]
    Q[0] = delta[0]/beta[0]
    for i in range(1, N-1):
        denom = beta[i] + alpha[i]*P[i-1]
        P[i] = -gamma[i]/denom
        Q[i] = (delta[i] - alpha[i]*Q[i-1])/denom

    # Зворотна прогонка
    for i in reversed(range(1, N)):
        if i == N-1:
            c[i] = Q[i-1]
        else:
            c[i] = P[i]*c[i+1] + Q[i]

    print("\nКоефіцієнти c:")
    for i,val in enumerate(c):
        print(f"c[{i}] = {val:.6f}")

    for i in range(N):
        b[i] = (y[i+1]-y[i])/h[i] - h[i]*(2*c[i]+c[i+1])/3
        d[i] = (c[i+1]-c[i])/(3*h[i])

    if N == 19: 
        print("\n=== Коефіцієнти для фінального сплайна (20 вузлів) ===")
        print("\nКоефіцієнти c:")
        for i, val in enumerate(c):
            print(f"c[{i}] = {val:.6f}")

        print("\nКоефіцієнти кубічних сплайнів:")
        for i in range(N):
            print(f"Інтервал {i}: a={a[i]:.3f}, b={b[i]:.6f}, c={c[i]:.6f}, d={d[i]:.9f}")

    return a, b, c[:-1], d



# 6. Побудова сплайнів

def build_spline_xy(x, y, label, target_x_points=None):
    h = np.diff(x)
    a, b, c, d = progonka(y, h)
    
    xx = np.linspace(x[0], x[-1], 500)
    yy = np.zeros_like(xx)
    for i, xi in enumerate(xx):
        idx = np.searchsorted(x, xi) - 1
        idx = max(0, min(idx, len(x)-2))
        dx = xi - x[idx]
        yy[i] = a[idx] + b[idx]*dx + c[idx]*dx**2 + d[idx]*dx**3
    
    plt.plot(xx, yy, label=label, linewidth=2)
    plt.plot(x, y, 'o', markersize=4)

    # значення в точках GPS для похибки ---
    if target_x_points is not None:
        y_pred = np.zeros_like(target_x_points)
        for i, xi in enumerate(target_x_points):
            idx = np.searchsorted(x, xi) - 1
            idx = max(0, min(idx, len(x)-2))
            dx = xi - x[idx]
            y_pred[i] = a[idx] + b[idx]*dx + c[idx]*dx**2 + d[idx]*dx**3
        return y_pred
    return None

# 7. Графіки

# 2 підграфіки
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

plt.sca(ax1)
plt.plot(distances, elevations, 'k.', label="Всі GPS точки (f(x))")

y_pred10 = build_spline_xy(x10, y10, "10 вузлів", distances)
y_pred15 = build_spline_xy(x15, y15, "15 вузлів", distances)
y_pred20 = build_spline_xy(x20, y20, "20 вузлів", distances)

ax1.set_ylabel("Висота, м")
ax1.set_title("Інтерполяція сплайнами y = f(x) та y_набл")
ax1.grid()
ax1.legend()

# графік похибки ε = |y - y_набл|
plt.sca(ax2)
ax2.plot(distances, np.abs(elevations - y_pred10), label="ε (10 вузлів)")
ax2.plot(distances, np.abs(elevations - y_pred15), label="ε (15 вузлів)")
ax2.plot(distances, np.abs(elevations - y_pred20), label="ε (20 вузлів)")

ax2.set_xlabel("Відстань, м")
ax2.set_ylabel("Похибка ε, м")
ax2.set_title("Графік похибки ε = |y - y_набл|")
ax2.grid()
ax2.legend()

plt.tight_layout()
plt.show()

# 8. Додатково

total_distance = distances[-1]
total_ascent = sum(max(elevations[i]-elevations[i-1],0) for i in range(1,n))
total_descent = sum(max(elevations[i-1]-elevations[i],0) for i in range(1,n))

print("\nЗагальна довжина маршруту (м):", round(total_distance,2))
print("Загальний набір висоти (м):", round(total_ascent,2))
print("Загальний спуск (м):", round(total_descent,2))

mass = 80
g = 9.81
energy = mass * g * total_ascent

print("Механічна робота (Дж):", round(energy,2))
print("Механічна робота (кДж):", round(energy/1000,2))
print("Енергія (ккал):", round(energy/4184,2))