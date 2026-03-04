import numpy
import matplotlib.pyplot as plot
import requests

class CubicSpline:

    @staticmethod
    def thomas_algorithm(alpha, beta, gamma, delta):
        n = len(alpha)
        a = [0]*n
        b = [0]*n
        x = [0]*n

        a[0] = -gamma[0]/beta[0]
        b[0] = delta[0]/beta[0]

        for i in range(1, n):
            denom = alpha[i]*a[i-1] + beta[i] #знаменник
            a[i] = -gamma[i]/denom
            b[i] = (delta[i] - alpha[i]*b[i-1]) / denom

        x[-1] = b[-1]
        for i in range(n-2, -1, -1):
            x[i] = a[i]*x[i+1] + b[i]

        return x

    def __init__(self, x_vals, y_vals):
        self.x = x_vals

        dx = [x_vals[i+1]-x_vals[i] for i in range(len(x_vals)-1)] #Різниця між відстанями двох сусідніх точок
        dy = [y_vals[i+1]-y_vals[i] for i in range(len(y_vals)-1)] #Різниця між висотами двох сусідніх точок

        n = len(dx) #кількість інтервалів між точками
        alpha = [0]*n #нижня
        beta  = [0]*n #центральна діагональ
        gamma = [0]*n #верхня
        delta = [0]*n

        beta[0] = 1

        for i in range(1, n):
            alpha[i] = dx[i-1]
            beta[i] = 2*(dx[i-1]+dx[i])
            gamma[i] = dx[i]
            delta[i] = 3*(dy[i]/dx[i] - dy[i-1]/dx[i-1])

        gamma[-1] = 0
        # с - друха похідна, кривизна
        # а - значення висоти точки
        self.c = CubicSpline.thomas_algorithm(alpha, beta, gamma, delta)
        self.a = y_vals[:-1]

        self.b = [] # нахил, перша похідна
        self.d = [] # зміна вигину на відрізку

        for i in range(len(self.a)-1):
            self.b.append(dy[i]/dx[i] - dx[i]/3*(self.c[i+1]+2*self.c[i]))
            self.d.append((self.c[i+1]-self.c[i])/(3*dx[i]))

        # з урахуванням того, що с = 0
        self.b.append(dy[-1]/dx[-1] - 2/3*dx[-1]*self.c[-1])
        self.d.append(-self.c[-1]/(3*dx[-1]))

    # видає висоту в будь якій точці маршруту
    def get_y(self, x):
        i = 0
        while i < len(self.x)-1 and x > self.x[i+1]:
            i += 1

        dx = x - self.x[i]
        return self.a[i] + self.b[i]*dx + self.c[i]*dx**2 + self.d[i]*dx**3


class GeographicPosition:

    EARTH_RADIUS_KILOMETERS = 6371

    def __init__(self, lat, lon, elev):
        self.latitude = lat
        self.longitude = lon
        self.elevation = elev

    @staticmethod
    def haversine_distance_kilometers(p1, p2):  #обчислення найкоротшої відстані між двома точками на сфері за їхньою широтою та довготою
        dlat = p2.latitude - p1.latitude
        dlon = p2.longitude - p1.longitude

        #на якій відстані знаходиться точка
        a = (1 - numpy.cos(dlat) + # зміна широти
             numpy.cos(p1.latitude)*numpy.cos(p2.latitude) *
             (1 - numpy.cos(dlon))) / 2 # зміна довготи

        return 2*numpy.arcsin(numpy.sqrt(a)) * GeographicPosition.EARTH_RADIUS_KILOMETERS


# дані

figure, axes = plot.subplots(1, 2, figsize=(12, 5), tight_layout=True)

url =  ("https://api.open-elevation.com/api/v1/lookup?"
    "locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|"
    "48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|"
    "48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|"
    "48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|"
    "48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|"
    "48.160580,24.500537|48.160250,24.500106")

data = requests.get(url).json()["results"]

points = [
    GeographicPosition(
        numpy.radians(p["latitude"]),
        numpy.radians(p["longitude"]),
        p["elevation"]
    )
    for p in data
]

# --- Кумулятивна відстань ---

distances = [0]
for i in range(1, len(points)):
    d = GeographicPosition.haversine_distance_kilometers(points[i], points[i-1])
    distances.append(distances[-1] + d)

# Аналіз маршруту

total_ascent = sum(
    max(0, points[i].elevation - points[i-1].elevation)
    for i in range(1, len(points))
)

total_descent = sum(
    max(0, points[i-1].elevation - points[i].elevation)
    for i in range(1, len(points))
)

print("Total distance:", distances[-1], "km")
print("Total ascent:", total_ascent, "m")
print("Total descent:", total_descent, "m")

print(f"\n{'№ точок':<8} | {'Кумулятивна відстань':<15}")
for i, d in enumerate(distances):
    print(f" {i+1:<2}      | {d:.4f} км")

# Побудова сплайнів

def plot_spline(spline, color, label):
    xs = numpy.linspace(distances[0], distances[-1], 200)
    ys = [spline.get_y(x) for x in xs]
    axes[0].plot(xs, ys, color=color, linewidth=1.2, label=label)


def plot_error(base, other, color, label):
    xs = numpy.linspace(distances[0], distances[-1], 200)
    error = [abs(base.get_y(x) - other.get_y(x)) for x in xs]
    axes[1].plot(xs, error, color=color, linewidth=1.2, label=label)


x_positions = distances
y_positions = [p.elevation/1000 for p in points]

base_spline = CubicSpline(x_positions, y_positions)
print("\nКоефіцієнти C:")
for c_val in base_spline.c:
    print(f"{float(c_val):.8f}")
plot_spline(base_spline, "black", "21 точка")

for n, color in [(10,"orange"), (15,"green"), (20,"blue")]:
    idx = numpy.linspace(0, len(x_positions)-1, n, dtype=int)
    x_sub = [x_positions[i] for i in idx]
    y_sub = [y_positions[i] for i in idx]

    spline_sub = CubicSpline(x_sub, y_sub)

    plot_spline(spline_sub, color, f"{n} точок")
    plot_error(base_spline, spline_sub, color, f"{n} точок")

    with open("points.txt", "w") as f:
        f.write("Lat,         Lon,        Elev\n")
        for i, p in enumerate(points):
            f.write(
                f"{numpy.degrees(p.latitude):.6f}, {numpy.degrees(p.longitude):.6f}, {p.elevation:.4f}\n")

axes[0].set_xlabel("Висота (км)")
axes[0].set_ylabel("Кумулятивна відстань (км)")
axes[0].set_title("Візуалізація кубічних сплайнів")
axes[0].legend()

axes[1].set_xlabel("Висота (км)")
axes[1].set_ylabel("Кумулятивна відстань (км)")
axes[1].set_title("Візуалізація похибки сплайнів")
axes[1].legend()

plot.show()