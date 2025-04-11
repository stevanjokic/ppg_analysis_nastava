import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Definicija funkcije sume dva Gaussa
def sum_of_gaussians(x, amp1, mean1, sigma1, amp2, mean2, sigma2):
    gauss1 = amp1 * np.exp(-(x - mean1)**2 / (2 * sigma1**2))
    gauss2 = amp2 * np.exp(-(x - mean2)**2 / (2 * sigma2**2))
    return gauss1 + gauss2

# Generisanje simuliranih podataka (primer)
# np.random.seed(42)  # Za reproduktivnost
# x_data = np.linspace(-10, 10, 200)
# y_data = (
#     3 * np.exp(-(x_data - 2)**2 / (2 * 1**2)) +  # Prvi Gauss
#     5 * np.exp(-(x_data + 3)**2 / (2 * 2**2)) +  # Drugi Gauss
#     np.random.normal(scale=0.2, size=x_data.size)  # Šum
# )
# initial_guess = [3, 2, 1, 5, -3, 2]  # Početne vrednosti za parametre

# 7395,7116,6818,6460,6019,5528,5056,4663,4369,4139,3921,3686,3460,3331,3380,3669,4190,4898,5713,6529,7258,7827,8229,8491,8656,8760,8824,8866,8897,8919,8912,8849,8713,8515,8283,8040,7795,7553,7330,7171,7119,7192,7349,7500,7561,7484,7292,7039,6779,6538,6309,6074,5816,5542,5260,4972,4658,4296,3895,3498,3161,2901,2676,2421,2097,1756,1475,1336,1374,1609,2054,2684,3408,4101,4661,5074,5373,5608,5806,5976,6114,6212,6258,6244,6167,6034,5863,5679,5495,5298,5062,4784,4509,4309,4235,4266,4344,4402,4399,4319

y_data = np.array([3704,3418,3127,2840,2564,2304,2052,1799,1535,1255,960,688,486,405,502,828,1390,2180,3174,4324,5551,6761,7855,8748,9388,9760,9914,9918,9833,9709,9578,9431,9250,9017,8719,8354,7942,7510,7086,6696,6360,6084,5867,5706,5596,5525,5480,5444,5403,5346,5266,5161,5033,4882,4709,4515,4301,4067,3814,3545,3262,2972,2681,2396,2122,1859,1602,1339,1064,772,474,219,64,69,300,797,1540,2491,3599,4785,5961,7052,7984,8704,9184,9434,9501,9443,9310,9139,8952,8740,8491,8194,7843,7444,7018,6590,6184,5825])
od = 12
do = 72
y_data = y_data[12:72]
y_data = (y_data - np.min(y_data)) / (np.max(y_data) - np.min(y_data))
x_data = np.array(list(range(len(y_data))))

# plt.figure(figsize=(12, 6))
# plt.plot(x_data, y_data, label="Podaci (observed)", color="blue")
# plt.show()

# Fitovanje podataka koristeći Levenberg-Marquardt algoritam (ugrađen u curve_fit)
# 3: Amplituda (𝐴 1) prvog Gaussa – određuje visinu vrha prvog Gaussa.
# 2: Centar (𝜇 1) prvog Gaussa – definiše poziciju prvog Gaussovog vrha na x-osi.
# 1: Standardna devijacija (𝜎 1) prvog Gaussa – označava širinu prvog Gaussa (veća vrednost znači širi oblik).
# 5: Amplituda (𝐴 2) drugog Gaussa – određuje visinu vrha drugog Gaussa.
# -3: Centar (𝜇 2) drugog Gaussa – pozicija vrha drugog Gaussa na x-osi.
# 2: Standardna devijacija (𝜎 2) drugog Gaussa – širina drugog Gaussa.

initial_guess = [1, 15, 7, .6, 35, 8]
y_initial_guess = sum_of_gaussians(x_data, *initial_guess)

params, params_covariance = curve_fit(
    sum_of_gaussians, x_data, y_data, p0=initial_guess, maxfev=5000
)

# Ispis rezultata fitovanja
print("Fitovani parametri:")
print(f"Amplitude i Centri: {params[:2]} , {params[3:5]}")
print(f"Širine (Sigma): {params[2]}, {params[5]}")

# Vizualizacija
# Prikaz rezultata
plt.figure(figsize=(12, 6))
plt.scatter(x_data, y_data, label="Podaci (observed)", color="blue", s=10)
plt.plot(x_data, sum_of_gaussians(x_data, *params), label="Fitovani rezultat", color="red", linewidth=2)
plt.plot(x_data, y_initial_guess, label="Početni Gaussi (initial guess)", color="green", linestyle="--", linewidth=2)
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Fitovanje Sume Dva Gaussa: Početni vs. Fitovani")
plt.grid()
plt.show()
