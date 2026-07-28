"""
This is the code for solving the $M=1$ single industry version of the model
"""
# Import packages
import os
import pickle
import time
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# Designate data and image directories
cur_dir = os.path.dirname(os.path.abspath(__file__))
# designate the data directory as "data/m1" and create those two directories
# if they do not exist
data_dir = os.path.join(cur_dir, "..", "data", "m1")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
images_dir = os.path.join(cur_dir, "..", "images", "m1")
if not os.path.exists(images_dir):
    os.makedirs(images_dir)


# Set parameters
years_per_period = 30
T = 15  # Guess for number of periods to steady-state
n1 = 1.0
n2 = 0.3
beta_annual = 0.96
beta = beta_annual ** (years_per_period)
print("Beta (per period):", beta)
sigma = 1.7
alpha_1 = 1.0
Z1 = 1.0
gamma1 = 0.35
delta1_annual = 0.05
delta1 = 1 - (1 - delta1_annual) ** (years_per_period)
print("Delta1 (per period):", delta1)
p1 = 1.0
c_min1 = 0.01
TPI_tol = 1e-9
TPI_max_iter = 1000
xi_tpi = 0.5  # Damping parameter for TPI iteration

# Define functions


def display_time(seconds):
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.5f}s"


def get_c_is(alpha_i, p_i, p, c_s, c_min_i):
    c_is = alpha_i * (p / p_i) * c_s + c_min_i

    return c_is


def get_c_s1(b2, w, p1, p, n1, c_min1):
    """
    Calculate the composite consumption of an young agent (s=1) from the budget
    constraint
    """
    c_s1 = (w / p) * n1 - (p1 / p) * c_min1 - (b2 / p)

    return c_s1


def get_c_s2(b2, w, r, p1, p, n2, c_min1):
    """
    Calculate the composite consumption of an old agent (s=2) from the budget
    constraint
    """
    c_s2 = ((1 + r) / p) * b2 + (w / p) * n2 - (p1 / p) * c_min1

    return c_s2


def get_Y(K, L, Z, gamma):
    Y = Z * (K ** gamma) * (L ** (1 - gamma))

    return Y


def get_L(n1, n2):
    L = n1 + n2

    return L


def get_w(K, L, p, Z, gamma):
    w = (1 - gamma) * p * Z * (K / L) ** gamma

    return w


def get_r(K, L, p, Z, gamma, delta):
    r = gamma * p * Z * (L / K) ** (1 - gamma) - delta

    return r


def get_b2_ss(w, r, n1, n2, c_min1, beta, sigma):
    numerator = (
        w * n1 - c_min1 -
        ((beta * (1 + r)) ** (-1 / sigma)) * (w * n2 - c_min1)
    )
    denominator = 1 + ((beta * (1 + r)) ** (-1 / sigma)) * (1 + r)
    b2_ss = numerator / denominator

    return b2_ss


def b2_ss_zerofunc(b2, *args):
    p, n1, n2, c_min1, Z, beta, sigma, gamma, delta = args
    L = get_L(n1, n2)
    K = b2
    w = get_w(K, L, p, Z, gamma)
    r = get_r(K, L, p, Z, gamma, delta)
    zerofunc = b2 - get_b2_ss(w, r, n1, n2, c_min1, beta, sigma)

    return zerofunc


def get_b2_tp(
    b2_0, w_path, r_path, p1_path, p_path, n1, n2, c_min1, beta, sigma
):
    T = len(w_path)
    b2_path = np.zeros(T)
    b2_path[0] = b2_0
    w_path_t = w_path[:-1]
    w_path_tp1 = w_path[1:]
    r_path_tp1 = r_path[1:]
    p1_path_t = p1_path[:-1]
    p1_path_tp1 = p1_path[1:]
    p_path_t = p_path[:-1]
    p_path_tp1 = p_path[1:]
    num_term1 = (w_path_t / p_path_t) * n1 - (p1_path_t / p_path_t) * c_min1
    num_term2 = (
        (
            (
                (beta * p_path_t * (1 + r_path_tp1)) / p_path_tp1
            ) ** (-1 / sigma)
        ) *
        ((w_path_tp1 / p_path_tp1) * n2 - (p1_path_tp1 / p_path_tp1) * c_min1)
    )
    denom = (
        (1 / p_path_t) + (
            (
                ((beta * p_path_t * (1 + r_path_tp1)) / p_path_tp1) **
                (-1 / sigma)
            ) * ((1 + r_path_tp1) / p_path_tp1)
        )
    )
    b2_path[1:] = (num_term1 - num_term2) / denom

    return b2_path


def series_plot(series, y_label, title=None, filename=None):
    plt.figure()
    T = len(series)
    per_vec = np.arange(1, T + 1)
    plt.plot(per_vec, series)
    plt.grid(True)
    # Add minor tick marks every 1 period only on the x-axis
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
    plt.ylabel(y_label)
    plt.xlabel(r"Period $t$")
    if title is not None:
        plt.title(title)
    if filename is not None:
        plt.savefig(os.path.join(images_dir, filename))
    plt.close()


# Solve for steady state equilibrium
start_time_ss = time.time()
sol_ss = opt.root_scalar(
    b2_ss_zerofunc, x0=0.01, args=(
        p1, n1, n2, c_min1, Z1, beta, sigma, gamma1, delta1
    )
)
compute_time_ss = time.time() - start_time_ss
print("Steady state compute time:", display_time(compute_time_ss))
print("sol_ss output:")
print(sol_ss)
b2_ss = sol_ss.root
print("Steady state b2:", b2_ss)
L1_ss = get_L(n1, n2)
print("Steady state L:", L1_ss)
K1_ss = b2_ss
print("Steady state K:", K1_ss)
Y1_ss = get_Y(K1_ss, L1_ss, Z1, gamma1)
w_ss = get_w(K1_ss, L1_ss, p1, Z1, gamma1)
print("Steady state w:", w_ss)
r_ss = get_r(K1_ss, L1_ss, p1, Z1, gamma1, delta1)
print("Steady state r:", r_ss)
c_s1_ss = get_c_s1(b2_ss, w_ss, p1, p1, n1, c_min1)
print("Steady state c1_1:", c_s1_ss)
c_s2_ss = get_c_s2(b2_ss, w_ss, r_ss, p1, p1, n2, c_min1)
print("Steady state c1_2:", c_s2_ss)
c1_1_ss = get_c_is(alpha_1, p1, p1, c_s1_ss, c_min1)
print("Steady state c1_1:", c1_1_ss)
c1_2_ss = get_c_is(alpha_1, p1, p1, c_s2_ss, c_min1)
print("Steady state c1_2:", c1_2_ss)
C1_ss = c1_1_ss + c1_2_ss
print("Steady state C1:", C1_ss)
I1_ss = delta1 * K1_ss
print("Steady state I1:", I1_ss)
resource_constraint_error_ss = Y1_ss - C1_ss - I1_ss
print("Steady state resource constraint error:", resource_constraint_error_ss)

ss_output = {
    "sol_ss": sol_ss,
    "b2_ss": b2_ss,
    "L1_ss": L1_ss,
    "K1_ss": K1_ss,
    "Y1_ss": Y1_ss,
    "w_ss": w_ss,
    "r_ss": r_ss,
    "c_s1_ss": c_s1_ss,
    "c_s2_ss": c_s2_ss,
    "c1_1_ss": c1_1_ss,
    "c1_2_ss": c1_2_ss,
    "C1_ss": C1_ss,
    "I1_ss": I1_ss,
    "resource_constraint_error_ss": resource_constraint_error_ss,
    "compute_time_ss": compute_time_ss
}
# Save ss_output dictionary as a pickle file "ss_output.pkl" in the data_dir
ss_output_file = os.path.join(data_dir, "ss_output.pkl")
with open(ss_output_file, "wb") as f:
    pickle.dump(ss_output, f)

# Solve for the transition path equilibrium
start_time_tpi = time.time()
K1_0 = 0.9 * K1_ss # Start with capital being 90% of the steady-state
b2_0 = 0.9 * b2_ss

# We know the equilibrium time path of aggregate labor and prices
L1_path = np.ones(T) * L1_ss
p1_path = np.ones(T)
p_path = np.ones(T)

# Guess a transition path for the aggregate capital stock
K1_path_init_guess = np.linspace(K1_0, K1_ss, T)
K1_path_init = K1_path_init_guess.copy()

TPI_dist = 100.0
TPI_iter = 0
print("")
print("Starting transition path equilibrium computation.")
while TPI_dist > TPI_tol and TPI_iter < TPI_max_iter:
    TPI_iter += 1
    w_path = get_w(K1_path_init, L1_path, p1_path, Z1, gamma1)
    r_path = get_r(K1_path_init, L1_path, p1_path, Z1, gamma1, delta1)
    b2_path = get_b2_tp(
        b2_0, w_path, r_path, p1_path, p_path, n1, n2, c_min1, beta, sigma
    )
    K1_path_new = b2_path.copy()
    TPI_dist = np.max(np.absolute(K1_path_new - K1_path_init))
    K1_path_init = xi_tpi * K1_path_new + (1 - xi_tpi) * K1_path_init
    print(f"Iteration {TPI_iter}: TPI_dist = {TPI_dist}")

K1_path = K1_path_init.copy()
b2_path = K1_path.copy()
w_path = get_w(K1_path, L1_path, p1_path, Z1, gamma1)
r_path = get_r(K1_path, L1_path, p1_path, Z1, gamma1, delta1)
I1_path = np.append(K1_path[1:], K1_ss) - (1 - delta1) * K1_path
Y1_path = get_Y(K1_path, L1_path, Z1, gamma1)
b2_path_tp1 = np.append(b2_path[1:], b2_ss)
c_s1_path = get_c_s1(b2_path_tp1, w_path, p1_path, p_path, n1, c_min1)
c_s2_path = get_c_s2(b2_path, w_path, r_path, p1_path, p_path, n2, c_min1)
c1_1_path = get_c_is(alpha_1, p1_path, p_path, c_s1_path, c_min1)
c1_2_path = get_c_is(alpha_1, p1_path, p_path, c_s2_path, c_min1)
C1_path = c1_1_path + c1_2_path

resource_constraint_error_path_tpi = Y1_path - C1_path - I1_path
print(
    "Maximum absolute resource constraint error:",
    np.max(np.absolute(resource_constraint_error_path_tpi))
)

compute_time_tpi = time.time() - start_time_tpi
print(
    "Transition path equilibrium compute time:",
    display_time(compute_time_tpi)
)

tpi_output = {
    "b2_path": b2_path,
    "L1_path": L1_path,
    "K1_path": K1_path,
    "Y1_path": Y1_path,
    "w_path": w_path,
    "r_path": r_path,
    "c_s1_path": c_s1_path,
    "c_s2_path": c_s2_path,
    "c1_1_path": c1_1_path,
    "c1_2_path": c1_2_path,
    "C1_path": C1_path,
    "I1_path": I1_path,
    "resource_constraint_error_path_tpi": resource_constraint_error_path_tpi,
    "compute_time_tpi": compute_time_tpi
}

# Save tpi_output dictionary as a pickle file "tpi_output.pkl" in the data_dir
tpi_output_file = os.path.join(data_dir, "tpi_output.pkl")
with open(tpi_output_file, "wb") as f:
    pickle.dump(tpi_output, f)

# Plot the following series
series_to_plot = [
    (b2_path, r"$b_{2,t+1}$", "Household savings", "b2_path.png"),
    (K1_path, r"$K_{1,t}$", "Aggregate capital", "K1_path.png"),
    (Y1_path, r"$Y_{1,t}$", "Output", "Y1_path.png"),
    (w_path, r"$w_t$", "Wage rate", "w_path.png"),
    (r_path, r"$r_t$", "Interest rate", "r_path.png"),
    (
        c_s1_path, r"$c_{s=1,t}$", "Household age 1 composite consumption",
        "c_s1_path.png"
    ),
    (
        c_s2_path, r"$c_{s=2,t}$", "Household age 2 composite consumption",
        "c_s2_path.png"
    ),
    (
        c1_1_path, r"$c_{1,1,t}$", "Household age 1 good 1 consumption",
        "c1_1_path.png"
    ),
    (
        c1_2_path, r"$c_{1,2,t}$", "Household age 2 good 1 consumption",
        "c1_2_path.png"
    ),
    (C1_path, r"$C_{1,t}$", "Aggregate consumption", "C1_path.png"),
    (I1_path, r"$I_{1,t}$", "Investment", "I1_path.png"),
    (
        resource_constraint_error_path_tpi, r"$Y_{1,t} - C_{1,t} - I_{1,t}$",
        "Resource constraint error", "resource_constraint_error_path_tpi.png"
    )
]
for inputs_tup in series_to_plot:
    series, y_label, title, filename = inputs_tup
    series_plot(series, y_label, title, filename)
