import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import lagrange
from scipy import optimize
def eval_function(x_prime, x, t):
    return np.exp(-x_prime) - x_prime - x**3 + 3*np.exp(-t**3)
def eval_prime(x_prime):
    return -np.exp(-x_prime) - 1
def newton_method(iter, x_prime, x, t): #running 100 iterations
    for _ in range(iter):
        x_prime = x_prime - eval_function(x_prime, x, t)/eval_prime(x_prime)
    return x_prime
def q3():
    x_0 = 1
    t_0 = 0
    t_f = 5
    N = 10**4
    h = (t_f-t_0)/(N-1)
    x_values = [x_0]
    t_values = [t_0]
    
    #forward euler's method
    while t_values[-1] + h <= t_f + h/2: #h/2 is tolerance in case of floating point errors, h is super small anyways so it should be fine
        x, t = x_values[-1], t_values[-1]
        x_prime = newton_method(100, 0, x, t)
        x_values.append(x + h * x_prime)
        t_values.append(t + h)
        
    print("used euler's method")
    # print("t values:", t_values)
    # print("x values:", x_values)
    print(f"first point: ({t_values[0]}, {x_values[0]})")
    print(f"last point: ({t_values[-1]}), ({x_values[-1]})")
    
    indices = [0]
    #finding which index has t=0,t=1,t=2,t=3,t=4,t=5, but need to do subtraction since floating point errors, so we get index with values of t closest to t=0, t=1, t=2, t=3, t=4, and t=5
    for i in range(1, 6):
        indices.append(np.argmin(np.abs(np.array(t_values) - i)))
    
    t_interpolate = [t_values[i] for i in indices]
    x_interpolate = [x_values[i] for i in indices]
    
    #create polynomial interpolation
    poly = lagrange(t_interpolate, x_interpolate)
    print("t-values taken for interpolation (has tolerance since floating point error):", t_interpolate)
    print("values of x taken for interpolation:", x_interpolate)
    print(poly)
    
    #extract and print polynomial coefficients
    coeff = poly.coef
    print("\npolynomial coefficients:")
    for i, c in enumerate(coeff):
        power = len(coeff)-i-1
        print(f"Coefficient of t^{power}: {c}")
    
    
    #t-exp fit
    def t_exp_model(t, alpha, beta):
        return 1 + alpha * t * np.exp(-beta*t)
    
    #creating a graph to fit to 6 points for the fitting of alpha beta
    params, _ = optimize.curve_fit(t_exp_model, t_interpolate, x_interpolate)
    alpha, beta = params
    print(f"\nt-exp fit parameters: alpha = {alpha:.6f}, beta = {beta:.6f}")
    
    #better t value for plots
    t_dense = np.linspace(0, 5, 1000)
    fig, ax = plt.subplots(3, 1, figsize=(8, 10))  # Smaller overall figure size
    plt.subplots_adjust(hspace=0.4)  # Reduce space between subplots
    #first plot for euler method
    ax[0].plot(t_values, x_values, 'b-')
    ax[0].set_title("(1) Solution of IVP using Forward Euler Method")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("x(t)")
    ax[0].grid(True)
    
    #second plot for interpolation
    ax[1].plot(t_values, x_values, 'b-', label='Euler solution', alpha=0.5)
    ax[1].plot(t_dense, [poly(t) for t in t_dense], 'r-', label='P_5(t) interpolation')
    ax[1].plot(t_interpolate, x_interpolate, 'ko', label='Sample points')
    ax[1].set_title("(2) Polynomial Interpolation P_5(t)")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("x(t)")
    ax[1].grid(True)
    ax[1].legend()
    
    #third plot for fitting
    ax[2].plot(t_values, x_values, 'b-', label='Euler solution', alpha=0.5)
    ax[2].plot(t_dense, [t_exp_model(t, alpha, beta) for t in t_dense], 'g-', label='t-exp fit')
    ax[2].plot(t_interpolate, x_interpolate, 'ko', label='Sample points')
    ax[2].set_title(f"(3) t-exp fit:")
    ax[2].set_xlabel("t")
    ax[2].set_ylabel("x(t)")
    ax[2].grid(True)
    ax[2].legend()
    
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    q3()