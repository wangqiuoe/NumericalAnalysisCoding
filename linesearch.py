# @author       :Qi Wang
# @date         :20201030

import numpy as np
import matplotlib.pyplot as plt

class LineSearch():

    # Implementation of algorithm in Numerical Analysis Chapter 3 Line Search Methods
    # optimize the Rosenbrock function in Chapter 2 (2.22)
    # the strongWolfe and zoom referred to https://gemfury.com/stream/python:scipy/-/content/optimize/linesearch.py

    def __init__(self, c1 = 1e-4, c2 = 0.9):
        self.c1 = c1
        self.c2 = c2
        self.max_strongWolfe_iter = 20
        self.max_zoom_iter = 20

    def gradient(self, x):
        x_1, x_2 = x[0], x[1]
        gradient = np.array([
            -400*x_1*x_2 + 400*np.power(x_1, 3) +2*x_1 - 2,
            200*x_2 - 200*np.power(x_1, 2)
        ])
        return gradient
    
    def hessian(self,x):
        x_1, x_2 = x[0], x[1]
        hessian = np.array([
            [-400*x_2 + 1200*np.power(x_1, 2) + 2, -400*x_1],
            [-400*x_1,                                  200]
        ])
        return hessian
    
    def func(self, x):
        x_1, x_2 = x[0], x[1]
        f_x = 100*np.power(x_2 - np.power(x_1, 2), 2) + np.power(1 - x_1, 2)
        return f_x
    
    def phi(self, x, alpha, p):
        return self.func(x + alpha*p)
    
    def derphi(self, x, alpha, p):
        return np.matmul(self.gradient(x+alpha*p), p)
    
    def strongWolfe(self, x, p, f=0, f_pre=0):
        #Algorithm 3.5 in page 60
        alpha_max = 1
        alpha_0 = 0
    
        #Initial alpha1
        derphi_0 = self.derphi(x, 0, p)
        if f_pre is not None and derphi_0 != 0:
            alpha_cur = min(1, 1.01*2*(f - f_pre)/derphi_0)
        else:
            alpha_cur = 1
        if alpha_cur < 0:
            alpha_cur = 1
        if alpha_cur == 0:
            # This shouldn't happen. Perhaps the increment has slipped below 
            #  machine precision?
            pass
    
        phi_0 =  f
        phi_pre = f
        alpha_pre = alpha_0
        for i in range(1, self.max_strongWolfe_iter+1):
            if alpha_cur == 0:
                break
            phi_cur = self.phi(x, alpha_cur, p)
            derphi_cur = self.derphi(x, alpha_cur, p)
            if (phi_cur > phi_0 + self.c1 * alpha_cur * derphi_0) or (phi_cur >= phi_pre and i > 1):
                alpha_star = self.zoom(x, p, alpha_pre, alpha_cur)
                break
            if abs(derphi_cur) <= -self.c2*derphi_0:
                alpha_star = alpha_cur
                break
            if derphi_cur >= 0:
                alpha_star = self.zoom(x, p, alpha_cur, alpha_pre)
                break
            alpha_pre = alpha_cur
            alpha_cur = 2*alpha_pre # increase by factor of two on each iteration ?? why?
            phi_pre = phi_cur
        else: # stopping test maxiter reached
            alpha_star = alpha_pre
            print('strongWolfe stopping because maxiter reached')    
        return alpha_star
        
    def interpolation(self, alpha_0, alpha_1):
        return (alpha_0 + alpha_1)/2 #TODO: need to implement quadratic, cubic, bisection

    def zoom(self, x, p, alpha_lo, alpha_hi):
        #Algorithm 3.6 in page 61
        
        phi_0 = self.phi(x, 0, p)
        derphi_0 = self.derphi(x, 0, p)

        for i in range(1, self.max_zoom_iter+1):
            alpha_j = self.interpolation(alpha_lo, alpha_hi)
            phi_j = self.phi(x, alpha_j, p)
            phi_lo = self.phi(x, alpha_lo, p)
            if (phi_j > phi_0 + self.c1 * alpha_j * derphi_0) or (phi_j >= phi_lo):
                alpha_hi = alpha_j
            else:
                derphi_j = self.derphi(x, alpha_j, p)
                if abs(derphi_j) <= -self.c2*derphi_0:
                    alpha_star = alpha_j
                    break
                if derphi_j*(alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo
                alpha_lo = alpha_j
        else: # stopping test maxiter reached
            alpha_star = None
        return alpha_star
    
    def backtrackingLS(self, x, p):
        #Algorithm 3.1 in page 37
        rho = 0.8
        c = 1e-4
        alpha = 1
        left_hand = self.func(x + alpha*p)
        right_hand = self.func(x) + c*alpha*np.matmul(self.gradient(x).T, p)
        while left_hand > right_hand:
            alpha *= rho
            left_hand = self.func(x + alpha*p)
            right_hand = self.func(x) + c*alpha*np.matmul(self.gradient(x).T, p)
            #print('alpha=%.4f, lhs=%.4f, rhs=%.4f' %(alpha, left_hand, right_hand))
        return alpha

    def BFGS_inv(self, x, x_pre, y, y_pre, B_inv_pre):
        delta_x = np.reshape(x - x_pre, [self.n, 1])
        delta_y = np.reshape(y - y_pre, [self.n, 1])
        rho = 1/(np.matmul(delta_y.T, delta_x))
        B_inv = np.matmul(np.eye(self.n) - rho* np.matmul(delta_x, delta_y.T), B_inv_pre)# equation 2.21 in page 25
        B_inv = np.matmul(B_inv, np.eye(self.n) - rho* np.matmul(delta_y, delta_x.T))
        B_inv =  B_inv + rho*np.matmul(delta_x, delta_x.T)
        p = -np.matmul(B_inv, self.gradient(x))
        return p, B_inv

    def main(self, x, max_iter = 5000, descent_method=1, ls_method = 1):
        y = self.func(x)
        y_pre = None
        x_pre = None
        curve_y = [y]
        i = 0
        error = 10
        x_str = '({x1:.4f},{x2:.4f})'.format(x1=x[0], x2 = x[1])
    
        print('{iter:>5s}{y:>10s}{alpha:>10s}{error:>10s}{x:>18s}'.format(iter='iter', y='f(x)', alpha='alpha', error='error', x='x'))
        print('{iter:>5d}{y:>10.2e}{alpha:>10.2e}{error:>10.2e}{x:>18s}'.format(iter=i, y=y, alpha=1, error=error, x=x_str))
        while error > 1e-10 and i < max_iter:
            i += 1

            if descent_method==1: #steepest descent descent_method
                p = -self.gradient(x)
            elif descent_method==2: #newton descent_method
                p = -np.matmul(np.linalg.inv(self.hessian(x)), self.gradient(x))
            elif descent_method==3: #quasi BFGS   #TODO how to initial B_inv_pre
                p, B_inv = self.BFGS_inv(x, x_pre, y, y_pre, B_inv_pre): 

            if ls_method == 1:
                alpha = self.backtrackingLS(x, p)
            elif ls_method == 2:
                alpha = self.strongWolfe(x, p, y, y_pre)

            x_pre = x
            x = x + alpha*p
            y_new = self.func(x)
            error = y - y_new
            y_pre = y
            y = y_new
            curve_y.append(y)
            x_str = '({x1:.4f},{x2:.4f})'.format(x1=x[0], x2 = x[1])
            print('{iter:>5d}{y:>10.2e}{alpha:>10.2e}{error:>10.2e}{x:>18s}'.format(iter=i, y=y, alpha=alpha, error=error, x=x_str))
        return curve_y
    
    
    def plot(self, y):
        n_iter = range(len(y))
        fig, ax = plt.subplots()
        line2, = ax.plot(n_iter, y, 'r+-', label='Using the dashes parameter')


if __name__ == '__main__':
    ls = LineSearch() 
    x = np.array([-1.2, 1])
    # descent_method: 1: Steepest Descent; 2: Newton; 3: quasi-BFGS
    # ls_method: 1: backtracking; 2: strong Wolfe
    y = ls.main(x, descent_method=2, ls_method = 1, max_iter=5000)

