import matplotlib.pyplot as plt
import os
import numpy as np
from zernike import RZern, FitZern
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# base feeder distance = 48.18
# base azimuth = 149.82384
# feeder_d_list의 실험의 azimuth 값이 다 149.82384


class Grasp:
    def __init__(self, base_path, file_names):
        self.base_path = base_path
        self.file_names = file_names
        self.theta_vals = [0, np.pi/4, np.pi/2, np.pi*3/4]
        self.total_rhos = []
        self.total_thetas = []
        self.total_w = []
        
        for id, file_name in enumerate(self.file_names):
            file_path = os.path.join(base_path[1], file_name)
        
            with open(file_path, 'r', encoding='UTF8') as f:
                lines = f.readlines()
                rhos = [] 
                ws = []
                thetas = []
                
                for idx, line in enumerate(lines):
                    if idx >= 156 and idx <=656 and line != '\n'  :
                    # if idx>=6 and line != '\n':
                        rho = float(line.split(',')[0])
                        print(rho)
                        w = float(line.split(',')[1].replace('\n',''))
                        rhos.append(rho)
                        ws.append(w)
                        thetas.append(self.theta_vals[id])
                # rhos = rhos[::50]
                # ws = ws[::50]
                # thetas = thetas[::50]
                
            self.total_rhos.append(rhos)
            self.total_thetas.append(thetas)
            self.total_w.append(ws)
        
        self.total_rhos = np.array(self.total_rhos)
        self.total_thetas = np.array(self.total_thetas)
        self.total_w = np.array(self.total_w)

        print("total_w", self.total_w)
        self.plot_3d(self.total_w)
        
        self.total_thetas[self.total_rhos<0] = self.total_thetas[self.total_rhos<0] + np.pi
        self.total_rhos = np.abs(self.total_rhos)/2.5
    
    def plot_3d(self, val):
        print("val",val)
        X, Y = self.total_rhos*np.cos(self.total_thetas), self.total_rhos*np.sin(self.total_thetas)
    
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(X.flatten(), Y.flatten(), val.flatten(), cmap=plt.cm.YlGnBu_r, edgecolor='none')
        # plt.show()
    
    def expression(self, n, m):
        z = self.total_w.copy()
        size = np.shape(z)
        
        for col_idx in range(size[0]):
            for row_idx in range(size[1]):
                if n==0 and m==0:
                    z[col_idx][row_idx] = 1
                elif n==1 and m==-1:
                    z[col_idx][row_idx] = 2*self.total_rhos[col_idx][row_idx]*np.sin(self.total_thetas[col_idx][row_idx])
                elif n==1 and m==1:
                    z[col_idx][row_idx] = 2*self.total_rhos[col_idx][row_idx]*np.cos(self.total_thetas[col_idx][row_idx])
                elif n==2 and m==-2:
                    z[col_idx][row_idx] = np.sqrt(6)*(self.total_rhos[col_idx][row_idx]**2)*np.sin(2*self.total_thetas[col_idx][row_idx])
                elif n==2 and m==0:
                    z[col_idx][row_idx] = np.sqrt(3)*(2*(self.total_rhos[col_idx][row_idx]**2)-1)
                elif n==2 and n==2:
                    z[col_idx][row_idx] = np.sqrt(6)*(self.total_rhos[col_idx][row_idx]**2)*np.cos(2*self.total_thetas[col_idx][row_idx])            
    
        z = np.array(z)
        return z
    
    def cal_alpha(self, z):
        alpha = 0
        _d_rho = 0.01
        
        for idx, w in enumerate(self.total_w.flatten()):
            alpha +=  z.flatten()[idx]* w * self.total_rhos.flatten()[idx] 
        alpha = alpha* _d_rho *np.pi/4 / np.pi
        print("alpha", alpha)

        return alpha

    """
    Zernike polynomials
    
    (n, m) 
    (0, 0) : Pistone 
    (1, -1) : Vetical tilt
    (1, 1) : Horizontal tilt
    (2, -2) : Oblique astigmatism
    (2, 0) : Defocus
    (2, 2) : Horizontal astigmatism
    """ 

if __name__=="__main__":
    base_path = ['./data/azimuth_exp', './data/distance_exp']
    file_names = ['feeder_d_48.18_0_deg_mag.txt', 'feeder_d_48.18_45_deg_mag.txt', 'feeder_d_48.18_90_deg_mag.txt', 'feeder_d_48.18_135_deg_mag.txt']
    # file_names = ['feeder_d_48.23_0_deg_mag.txt', 'feeder_d_48.23_45_deg_mag.txt', 'feeder_d_48.23_90_deg_mag.txt', 'feeder_d_48.23_135_deg_mag.txt']
    # file_names = ['feeder_d_48.33_0_deg_mag.txt', 'feeder_d_48.33_45_deg_mag.txt', 'feeder_d_48.33_90_deg_mag.txt', 'feeder_d_48.33_135_deg_mag.txt']

    # file_names = ['feeder_d_48.15_0_deg_phase.txt', 'feeder_d_48.15_45_deg_phase.txt', 'feeder_d_48.15_90_deg_phase.txt', 'feeder_d_48.15_135_deg_phase.txt']    
    # file_names = ['feeder_d_48.18_0_deg_phase.txt', 'feeder_d_48.18_45_deg_phase.txt', 'feeder_d_48.18_90_deg_phase.txt', 'feeder_d_48.18_135_deg_phase.txt']    
    # file_names = ['feeder_d_48.23_0_deg_phase.txt', 'feeder_d_48.23_45_deg_phase.txt', 'feeder_d_48.23_90_deg_phase.txt', 'feeder_d_48.23_135_deg_phase.txt']
    # file_names = ['feeder_d_48.33_0_deg_phase.txt', 'feeder_d_48.33_45_deg_phase.txt', 'feeder_d_48.33_90_deg_phase.txt', 'feeder_d_48.33_135_deg_phase.txt']


    grasp = Grasp(base_path, file_names)
    
    z_pistone = grasp.expression(0, 0)
    alpha_pisonte = grasp.cal_alpha(z_pistone)
    
    z_h_tilt = grasp.expression(1, 1)
    alpha_h_tilt = grasp.cal_alpha(z_h_tilt)
    
    z_v_tilt = grasp.expression(1, -1)
    alpha_v_tilt = grasp.cal_alpha(z_v_tilt)
    
    z_oa = grasp.expression(2, -2)
    alpha_oa = grasp.cal_alpha(z_oa)
    
    z_defocus = grasp.expression(2, 0)
    alpha_defocus = grasp.cal_alpha(z_defocus)
    
    z_ha = grasp.expression(2, 2)
    alpha_ha = grasp.cal_alpha(z_ha)
    
    estimation = z_pistone*z_pistone + z_h_tilt*alpha_h_tilt + z_v_tilt*alpha_v_tilt + z_oa*alpha_oa + z_defocus*alpha_defocus + z_ha*alpha_ha
    estimation = estimation
    print("estimation", estimation)
    grasp.plot_3d(estimation)
    
    plt.show()
    
    # grasp.plot_3d(z_h_tilt*alpha_h_tilt)
    
    

    
    
    
    
    
    
    
"""
# def expression(n, m, rhos, thetas, total_w):
#     z=total_w.copy()
#     size = np.shape(z)
#     # print("orginal matrix", total_w)
    
#     for col_idx in range(size[0]):
#         for row_idx in range(size[1]):
#             if n==0 and m==0:
#                 z[col_idx][row_idx] = 1
#             elif n==1 and m==-1:
#                 z[col_idx][row_idx] = 2*rhos[col_idx][row_idx]*np.sin(thetas[col_idx][row_idx])
#             elif n==1 and m==1:
#                 z[col_idx][row_idx] = 2*rhos[col_idx][row_idx]*np.cos(thetas[col_idx][row_idx])
#             elif n==2 and m==-2:
#                 z[col_idx][row_idx] = np.sqrt(6)*(rhos[col_idx][row_idx]**2)*np.sin(2*thetas[col_idx][row_idx])
#             elif n==2 and m==0:
#                 z[col_idx][row_idx] = np.sqrt(3)*(2*(rhos[col_idx][row_idx]**2)-1)
#             elif n==2 and n==2:
#                 z[col_idx][row_idx] = np.sqrt(6)*(rhos[col_idx][row_idx]**2)*np.cos(2*thetas[col_idx][row_idx])
    
#     # X, Y = rhos*np.cos(thetas), rhos*np.sin(thetas)
#     # fig = plt.figure()
#     # ax = fig.add_subplot(projection='3d')
#     # ax.plot_trisurf(X.flatten(), Y.flatten(), matrix.flatten(), cmap=plt.cm.YlGnBu_r, edgecolor='none')
#     # plt.title(f'n, m {n, m}')
    
#     matrix = np.array(matrix)
#     # print("matrix",matrix)
    
#     return matrix


    # w = np.array([])
    # theta_vals = [0, np.pi/4, np.pi/2, np.pi*3/4]
    # total_rhos = []
    # total_thetas = []
    # total_w = []
    # for id, file_name in enumerate(file_names):
    #     file_path = os.path.join(base_path[1], file_name)
        
    #     with open(file_path, 'r', encoding='UTF8') as f:
    #         lines = f.readlines()
    #         rhos = [] 
    #         ws = []
    #         thetas = []
            
    #         for idx, line in enumerate(lines):
    #             if idx >= 156 and idx <=686 and line != '\n'  :
    #                 rho = float(line.split(',')[0])
    #                 w = float(line.split(',')[1].replace('\n',''))
    #                 rhos.append(rho)
    #                 ws.append(w)
    #                 thetas.append(theta_vals[id])
    #         rhos = rhos[::50]
    #         ws = ws[::50]
    #         thetas = thetas[::50]
            
    #     total_rhos.append(rhos)
    #     total_thetas.append(thetas)
    #     total_w.append(ws)
    
    # total_rhos = np.array(total_rhos)
    # total_thetas = np.array(total_thetas)
    # total_w = np.array(total_w)
    
    # total_thetas[total_rhos<0] = total_thetas[total_rhos<0] + np.pi
    # total_rhos = np.abs(total_rhos)/2.5
    
    # print(total_rhos)
    # print(total_thetas)
    
    
    # X, Y = total_rhos*np.cos(total_thetas), total_rhos*np.sin(total_thetas)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.plot_trisurf(X.flatten(), Y.flatten(), total_w.flatten(), cmap=plt.cm.YlGnBu_r, edgecolor='none')
    # # plt.show()
    
    # Z_pistone = np.ones_like(total_w)
    # # alpha_pistone = np.transpose(Z_pistone.flatten()) @ total_w.flatten() * Z_pistone.flatten() * 0.2 *np.pi/4 / np.pi
    # alpha_pistone = 0 
    # Z_pistone = expression(0, 0, total_rhos, total_thetas, total_w=total_w)
    # # print("Z_pistone",Z_pistone)
    # for idx, w in enumerate(total_w.flatten()):
    #     alpha_pistone += 1 * w * total_rhos.flatten()[idx] 
    # alpha_pistone = alpha_pistone* 0.2 *np.pi/4 / np.pi
            
    # # print("before", total_w)
    # Z_v_tilt = expression(1, -1, total_rhos, total_thetas, total_w=total_w)
    # # alpha_v_tilt = np.transpose(Z_v_tilt.flatten()) @ total_w.flatten()
    # alpha_v_tilt = 0
    # Z_v_tilt_flat = Z_v_tilt.flatten()
    # for idx, w in enumerate(total_w.flatten()):
    #     alpha_v_tilt +=  Z_v_tilt_flat[idx]* w * total_rhos.flatten()[idx] 
    # alpha_v_tilt = alpha_v_tilt* 0.2 *np.pi/4 / np.pi
    # print("alpha_v_tilt", alpha_v_tilt)
    
    
    # # print("after", total_w)
    # Z_h_tilt = expression(1, 1, total_rhos, total_thetas, total_w=total_w)
    # # alpha_h_tilt = np.transpose(Z_h_tilt.flatten()) @ total_w.flatten()
    # alpha_h_tilt = 0
    # Z_h_tilt_flat = Z_h_tilt.flatten()
    # for idx, w in enumerate(total_w.flatten()):
    #     alpha_h_tilt +=  Z_h_tilt_flat[idx]* w * total_rhos.flatten()[idx] 
    # alpha_h_tilt = alpha_h_tilt* 0.2 *np.pi/4 / np.pi
    # print("alpha_h_tilt", alpha_h_tilt)
    
    # # Z_oblique_astigmatism = expression(2, -2, total_rhos, total_thetas, total_w=total_w)
    # # # alpha_oa = np.transpose(Z_oblique_astigmatism.flatten()) @ total_w.flatten()
    # # alpha_oa = 0
    # # Z_oblique_astigmatism_flat = Z_oblique_astigmatism.flatten()
    # # for idx, w in enumerate(total_w.flatten()):
    # #     alpha_oa +=  Z_oblique_astigmatism_flat[idx]* w * total_rhos.flatten()[idx] 
    # # alpha_oa = alpha_oa* 0.2 *np.pi/4 / np.pi
    # # print("alpha_oa", alpha_oa)
    
    # Z_defocus = expression(2, 0, total_rhos, total_thetas, total_w=total_w)
    # # alpha_defocus = np.transpose(Z_defocus.flatten()) @ total_w.flatten()
    # alpha_defocus = 0
    # Z_defocus_flat = Z_defocus.flatten()
    # for idx, w in enumerate(total_w.flatten()):
    #     alpha_defocus +=  Z_defocus_flat[idx]* w * total_rhos.flatten()[idx] 
    # alpha_defocus = alpha_defocus* 0.2 *np.pi/4 / np.pi    
    # print("alpha_defocus", alpha_defocus)
    
    # # Z_horizontal_astigmatism = expression(2, 2, total_rhos, total_thetas, total_w=total_w)
    # # # alpha_ha = np.transpose(Z_horizontal_astigmatism.flatten()) @ total_w.flatten()
    # # alpha_ha = 0
    # # Z_horizontal_astigmatism_flat = Z_horizontal_astigmatism.flatten()
    # # for idx, w in enumerate(total_w.flatten()):
    # #     alpha_ha +=  Z_horizontal_astigmatism_flat[idx]* w * total_rhos.flatten()[idx] 
    # # alpha_ha = alpha_ha* 0.2 *np.pi/4 / np.pi        
    # # print("alpha_ha", alpha_ha)
    
    
    # # order_sum = alpha_pistone*Z_pistone+alpha_v_tilt*Z_v_tilt + alpha_h_tilt*Z_h_tilt + alpha_oa*Z_oblique_astigmatism + alpha_defocus*Z_defocus + alpha_ha*Z_horizontal_astigmatism
    # order_sum = alpha_pistone*Z_pistone+alpha_v_tilt*Z_v_tilt + alpha_h_tilt*Z_h_tilt + alpha_defocus*Z_defocus 
    # X, Y = total_rhos*np.cos(total_thetas), total_rhos*np.sin(total_thetas)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.plot_trisurf(X.flatten(), Y.flatten(), order_sum.flatten(), cmap=plt.cm.YlGnBu_r, edgecolor='none')
    # ax.set_title("sum")
    # plt.show()
"""
    
    
    
    
    
    
    
    

