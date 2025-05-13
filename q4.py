import numpy as np
import matplotlib.pyplot as plt

def solve_bvp(evaluate, a, b, y_a, y_b, h):
    #set up of arrays
    x_values = np.arange(a, b + h/2, h)
    n = len(x_values)
    #y'(a)=0 case   
    y1 = np.zeros(n)
    y1_prime = np.zeros(n)
    #y'(a)=1 case
    y2 = np.zeros(n)
    y2_prime = np.zeros(n)
    y1[0] = y_a
    y1_prime[0] = 0
    y2[0] = y_a
    y2_prime[0] = 1
    
    #midpoint method
    for i in range(n-1):
        x = x_values[i]
        
        #y'(a) = 0 IVP
        #get derivative at this point
        dy1 = y1_prime[i]
        dyp1 = evaluate(x, y1[i], y1_prime[i])
        #get midpoint value
        y_mid = y1[i] + 0.5*h*dy1
        yp_mid = y1_prime[i] + 0.5*h*dyp1
        x_mid = x + 0.5*h
        #update new midpoint derivatives and values
        y1[i+1] = y1[i] + h * yp_mid
        y1_prime[i+1] = y1_prime[i] + h * evaluate(x_mid, y_mid, yp_mid)
        
        #y'(a) = 1 IVP
        #get derivative at this point
        dy2 = y2_prime[i]
        dyp2 = evaluate(x, y2[i], y2_prime[i])
        #get midpoint value
        y_mid = y2[i] + 0.5*h*dy2
        yp_mid = y2_prime[i] + 0.5*h*dyp2
        x_mid = x + 0.5*h
        #update new midpoint derivatives and values
        y2[i+1] = y2[i] + h * yp_mid
        y2_prime[i+1] = y2_prime[i] + h * evaluate(x_mid, y_mid, yp_mid)
    
    #finding shooting parameter using the formula
    print("y1:", y1)
    print("y1_prime", y1_prime)
    print("y2", y2)
    print("y2_prime", y2_prime)
    s = (y_b - y1[-1]) / (y2[-1] - y1[-1])
    
    #combine solutions linearly, gets actual array of points that the graph follows
    y_values = y1 + s * (y2 - y1)
    return x_values, y_values, y1, y2, s


def bvp():
    def evaluate(x, y, y_prime):
        return -x * y  # y'' = -xy
    a = 0      # left boundary
    b = 2      # right boundary
    y_a = 1    # y(a) = 1
    y_b = 2    # y(b) = 3
    
    #solve the BVP
    x_values, y_values, y1, y2, s = solve_bvp(evaluate, a, b, y_a, y_b, h=2/200)
    print('shooting value: ', s)
    print("Each value at each step:")
    print(f"{'x'} \t {'y(x)'}")
    
    #print results
    for i in range(len(x_values)):
        print(f"{x_values[i]} \t {y_values[i]}")

if __name__ == "__main__":
    bvp()