import matplotlib.pyplot as plt
import os
import numpy as np
from zernike import RZern, FitZern
import matplotlib.pyplot as plt

# base feeder distance = 48.18
# base azimuth = 149.82384
# feeder_d_list의 실험의 azimuth 값이 다 149.82384

base_path = ['./data/azimuth_exp', './data/distance_exp']

files = os.listdir(base_path[0])

file_path = os.path.join(base_path[0], files[0])

x_list = list(np.linspace(-4, 4, 801))
# print(x_list.index(-2.5), x_list.index(2.5))



y = []

with open(file_path, 'r', encoding='UTF8') as f:
    lines = f.readlines()
    
    for idx, line in enumerate(lines):
        if idx >= 156 and idx <=656 and line != '\n'  :
            # print(line.split(',')[1].replace('\n',''))
            x_value = float(line.split(',')[0])
            # print(x_value)
            y_value = float(line.split(',')[1].replace('\n',''))
            y.append(y_value)
    
    y = y[::40]
    
            
x_list = list(np.linspace(-2.5, 2.5, 501))
x_list = x_list[::40]



pol = RZern(3)
# rho = L, theta = K
L, K = 13, 9
ip = FitZern(pol, L, K)
# print(ip.rho_j[0], ip.theta_i)

rho_j = np.array(x_list)
theta_i = np.array([0.0, 0.7853981634, 1.570796327, 2.35619449, 3.141592654, 3.926990817, 4.71238898, 5.497787144, 6.283185307])
pol.make_pol_grid(rho_j, theta_i)
c_true = np.random.normal(size=pol.nk)


print(np.random.normal(size=pol.nk))
# print(c_true)

# Phi = pol.eval_grid(c_true)
# c_hat = ip.fit(Phi)
# R = np.zeros((pol.nk, 3))
# R[:, 0] = c_true
# R[:, 1] = c_hat
# R[:, 2] = np.abs(c_true - c_hat)
# print(R)
# np.linalg.norm(c_true - c_hat)/np.linalg.norm(c_true)





# for file in files:
#     file_path = os.path.join(base_path[0], file)
#     y = []
#     with open(file_path, 'r', encoding='UTF8') as f:
#         lines = f.readlines()
        
#         for idx, line in enumerate(lines):
#             if idx >= 6 and line != '\n':
#                 # print(line.split(',')[1].replace('\n',''))
#                 y_value = float(line.split(',')[1].replace('\n',''))
#                 if y_value >= -2.5 and y_value <= 2.5:
#                     y.append(y_value)


            


#     plt.figure()
#     plt.subplot(2, 2, 1)
#     plt.plot(x_list, y, 'r')
#     plt.show()
    

# ddx = np.linspace(-4.0, 4.0, K)
# ddy = np.linspace(-4.0, 4.0, L)
# xv, yv = np.meshgrid(ddx, ddy)
# cart.make_cart_grid(xv, yv)

# c = np.zeros(cart.nk)
# plt.figure(1)
# for i in range(1, 10):
#     plt.subplot(3, 3, i)
#     c *= 0.0
#     c[i] = 1.0
#     Phi = cart.eval_grid(c, matrix=True)
#     plt.imshow(Phi, origin='lower', extent=(-1, 1, -1, 1))
#     plt.axis('off')

# plt.show()