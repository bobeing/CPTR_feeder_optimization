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
            file_path = os.path.join(base_path[4], file_name)
        
            with open(file_path, 'r', encoding='UTF8') as f:
                lines = f.readlines()
                rhos = [] 
                ws = []
                thetas = []
                
                
                for idx, line in enumerate(lines):
                    if idx>=6 and line != '\n':
                        rho = float(line.split(',')[0])
                        w = float(line.split(',')[1].replace('\n',''))
                        rhos.append(rho)
                        ws.append(w)
                        thetas.append(self.theta_vals[id])
                        
                
                # -4m ~ 4m 를 측정하고 샘플개수가 801개일 때
                # for idx, line in enumerate(lines):
                #     if idx >= 156 and idx <=656 and line != '\n'  :
                #     # if idx>=6 and line != '\n':
                #         rho = float(line.split(',')[0])
                #         w = float(line.split(',')[1].replace('\n',''))
                #         rhos.append(rho)
                #         ws.append(w)
                #         thetas.append(self.theta_vals[id])
                # rhos = rhos[::50]
                # ws = ws[::50]
                # thetas = thetas[::50]
                
            self.total_rhos.append(rhos)
            self.total_thetas.append(thetas)
            self.total_w.append(ws)
        
        self.total_rhos = np.array(self.total_rhos)
        self.total_thetas = np.array(self.total_thetas)
        self.total_w = np.array(self.total_w)

        self.plot_3d(self.total_w)
        
        self.total_thetas[self.total_rhos<0] = self.total_thetas[self.total_rhos<0] + np.pi
        self.total_rhos = np.abs(self.total_rhos)/2.5
    
    def plot_3d(self, val):
        X, Y = self.total_rhos*np.cos(self.total_thetas), self.total_rhos*np.sin(self.total_thetas)
    
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(X.flatten(), Y.flatten(), val.flatten(), cmap=plt.cm.YlGnBu_r, edgecolor='none')
        ax.set_zlim(-160, -120)
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
    base_path = ['./data/azimuth_exp', './data/distance_exp', './data/feeder_distance_10mm', './data/feeder_distance_10mm_resample', './data/feeder_distance_final']
    # file_names = ['feeder_d_48.18_0_deg_mag.txt', 'feeder_d_48.18_45_deg_mag.txt', 'feeder_d_48.18_90_deg_mag.txt', 'feeder_d_48.18_135_deg_mag.txt']
    # file_names = ['feeder_d_48.23_0_deg_mag.txt', 'feeder_d_48.23_45_deg_mag.txt', 'feeder_d_48.23_90_deg_mag.txt', 'feeder_d_48.23_135_deg_mag.txt']
    # file_names = ['feeder_d_48.33_0_deg_mag.txt', 'feeder_d_48.33_45_deg_mag.txt', 'feeder_d_48.33_90_deg_mag.txt', 'feeder_d_48.33_135_deg_mag.txt']

    
    # file_names = ['feeder_d_48.14_0_deg_phase.txt', 'feeder_d_48.14_45_deg_phase.txt', 'feeder_d_48.14_90_deg_phase.txt', 'feeder_d_48.14_135_deg_phase.txt']
    # file_names = ['feeder_d_48.15_0_deg_phase.txt', 'feeder_d_48.15_45_deg_phase.txt', 'feeder_d_48.15_90_deg_phase.txt', 'feeder_d_48.15_135_deg_phase.txt']    
    # file_names = ['feeder_d_48.16_0_deg_phase.txt', 'feeder_d_48.16_45_deg_phase.txt', 'feeder_d_48.16_90_deg_phase.txt', 'feeder_d_48.16_135_deg_phase.txt']
    # file_names = ['feeder_d_48.17_0_deg_phase.txt', 'feeder_d_48.17_45_deg_phase.txt', 'feeder_d_48.17_90_deg_phase.txt', 'feeder_d_48.17_135_deg_phase.txt']
    # file_names = ['feeder_d_48.18_0_deg_phase.txt', 'feeder_d_48.18_45_deg_phase.txt', 'feeder_d_48.18_90_deg_phase.txt', 'feeder_d_48.18_135_deg_phase.txt']    
    # file_names = ['feeder_d_48.19_0_deg_phase.txt', 'feeder_d_48.19_45_deg_phase.txt', 'feeder_d_48.19_90_deg_phase.txt', 'feeder_d_48.19_135_deg_phase.txt']
    # file_names = ['feeder_d_48.20_0_deg_phase.txt', 'feeder_d_48.20_45_deg_phase.txt', 'feeder_d_48.20_90_deg_phase.txt', 'feeder_d_48.20_135_deg_phase.txt']
    # file_names = ['feeder_d_48.21_0_deg_phase.txt', 'feeder_d_48.21_45_deg_phase.txt', 'feeder_d_48.21_90_deg_phase.txt', 'feeder_d_48.21_135_deg_phase.txt']
    # file_names = ['feeder_d_48.23_0_deg_phase.txt', 'feeder_d_48.23_45_deg_phase.txt', 'feeder_d_48.23_90_deg_phase.txt', 'feeder_d_48.23_135_deg_phase.txt']
    # file_names = ['feeder_d_48.33_0_deg_phase.txt', 'feeder_d_48.33_45_deg_phase.txt', 'feeder_d_48.33_90_deg_phase.txt', 'feeder_d_48.33_135_deg_phase.txt']
    
    file_dict = {
        '48.08': ['feeder_d_48.08_0_deg_phase.txt', 'feeder_d_48.08_45_deg_phase.txt', 'feeder_d_48.08_90_deg_phase.txt', 'feeder_d_48.08_135_deg_phase.txt'],
        '48.14': ['feeder_d_48.14_0_deg_phase.txt', 'feeder_d_48.14_45_deg_phase.txt', 'feeder_d_48.14_90_deg_phase.txt', 'feeder_d_48.14_135_deg_phase.txt'],
        '48.15': ['feeder_d_48.15_0_deg_phase.txt', 'feeder_d_48.15_45_deg_phase.txt', 'feeder_d_48.15_90_deg_phase.txt', 'feeder_d_48.15_135_deg_phase.txt'],
        '48.16': ['feeder_d_48.16_0_deg_phase.txt', 'feeder_d_48.16_45_deg_phase.txt', 'feeder_d_48.16_90_deg_phase.txt', 'feeder_d_48.16_135_deg_phase.txt'],
        '48.17': ['feeder_d_48.17_0_deg_phase.txt', 'feeder_d_48.17_45_deg_phase.txt', 'feeder_d_48.17_90_deg_phase.txt', 'feeder_d_48.17_135_deg_phase.txt'],
        '48.18': ['feeder_d_48.18_0_deg_phase.txt', 'feeder_d_48.18_45_deg_phase.txt', 'feeder_d_48.18_90_deg_phase.txt', 'feeder_d_48.18_135_deg_phase.txt'],
        '48.19': ['feeder_d_48.19_0_deg_phase.txt', 'feeder_d_48.19_45_deg_phase.txt', 'feeder_d_48.19_90_deg_phase.txt', 'feeder_d_48.19_135_deg_phase.txt'],
        '48.20': ['feeder_d_48.20_0_deg_phase.txt', 'feeder_d_48.20_45_deg_phase.txt', 'feeder_d_48.20_90_deg_phase.txt', 'feeder_d_48.20_135_deg_phase.txt'],
        '48.21': ['feeder_d_48.21_0_deg_phase.txt', 'feeder_d_48.21_45_deg_phase.txt', 'feeder_d_48.21_90_deg_phase.txt', 'feeder_d_48.21_135_deg_phase.txt'],
        '48.28': ['feeder_d_48.28_0_deg_phase.txt', 'feeder_d_48.28_45_deg_phase.txt', 'feeder_d_48.28_90_deg_phase.txt', 'feeder_d_48.28_135_deg_phase.txt'],
        '48.106788': ['feeder_d_48.106788_0_deg_phase.txt', 'feeder_d_48.106788_45_deg_phase.txt', 'feeder_d_48.106788_90_deg_phase.txt', 'feeder_d_48.106788_135_deg_phase.txt'],
        # '48.2452':['feeder_d_48.2452_0_deg_phase.txt', 'feeder_d_48.2452_45_deg_phase.txt', 'feeder_d_48.2452_90_deg_phase.txt', 'feeder_d_48.2452_135_deg_phase.txt'],
        # '48.12332':['feeder_d_48.12332_0_deg_phase.txt', 'feeder_d_48.12332_45_deg_phase.txt', 'feeder_d_48.12332_90_deg_phase.txt', 'feeder_d_48.12332_135_deg_phase.txt'],
        # '48.20923154':['feeder_d_48.20923154_0_deg_phase.txt', 'feeder_d_48.20923154_45_deg_phase.txt', 'feeder_d_48.20923154_90_deg_phase.txt', 'feeder_d_48.20923154_135_deg_phase.txt'],
        # '48.1190345':['feeder_d_48.1190345_0_deg_phase.txt', 'feeder_d_48.1190345_45_deg_phase.txt', 'feeder_d_48.1190345_90_deg_phase.txt', 'feeder_d_48.1190345_135_deg_phase.txt'],
        # '48.22236639':['feeder_d_48.22236639_0_deg_phase.txt', 'feeder_d_48.22236639_45_deg_phase.txt', 'feeder_d_48.22236639_90_deg_phase.txt', 'feeder_d_48.22236639_135_deg_phase.txt']
        '48.20494164': ['feeder_d_48.20494164_0_deg_phase.txt', 'feeder_d_48.20494164_45_deg_phase.txt', 'feeder_d_48.20494164_90_deg_phase.txt', 'feeder_d_48.20494164_135_deg_phase.txt'],
        '48.16653184': ['feeder_d_48.16653184_0_deg_phase.txt', 'feeder_d_48.16653184_45_deg_phase.txt', 'feeder_d_48.16653184_90_deg_phase.txt', 'feeder_d_48.16653184_135_deg_phase.txt'],
        '48.11249312': ['feeder_d_48.11249312_0_deg_phase.txt', 'feeder_d_48.11249312_45_deg_phase.txt', 'feeder_d_48.11249312_90_deg_phase.txt', 'feeder_d_48.11249312_135_deg_phase.txt'],
        '48.18863725': ['feeder_d_48.18863725_0_deg_phase.txt', 'feeder_d_48.18863725_45_deg_phase.txt', 'feeder_d_48.18863725_90_deg_phase.txt', 'feeder_d_48.18863725_135_deg_phase.txt'],
        '48.1597859': ['feeder_d_48.1597859_0_deg_phase.txt', 'feeder_d_48.1597859_45_deg_phase.txt', 'feeder_d_48.1597859_90_deg_phase.txt', 'feeder_d_48.1597859_135_deg_phase.txt'],
        '48.24960251': ['feeder_d_48.24960251_0_deg_phase.txt', 'feeder_d_48.24960251_45_deg_phase.txt', 'feeder_d_48.24960251_90_deg_phase.txt', 'feeder_d_48.24960251_135_deg_phase.txt'],
        '48.10903039': ['feeder_d_48.10903039_0_deg_phase.txt', 'feeder_d_48.10903039_45_deg_phase.txt', 'feeder_d_48.10903039_90_deg_phase.txt', 'feeder_d_48.10903039_135_deg_phase.txt'],
        '48.16331369': ['feeder_d_48.16331369_0_deg_phase.txt', 'feeder_d_48.16331369_45_deg_phase.txt', 'feeder_d_48.16331369_90_deg_phase.txt', 'feeder_d_48.16331369_135_deg_phase.txt'],
        '48.08012225': ['feeder_d_48.08012225_0_deg_phase.txt', 'feeder_d_48.08012225_45_deg_phase.txt', 'feeder_d_48.08012225_90_deg_phase.txt', 'feeder_d_48.08012225_135_deg_phase.txt']
    }
    
    # key_list = ['48.14','48.15', '48.16', '48.17', '48.18', '48.19', '48.20', '48.21']
    key_list = ['48.19']
    
    n_m_list = [(0,0), (1,-1), (1,1), (2,0)]
    x_list = []
    # x_list = np.array([])
    for idx, key in enumerate(key_list):
        grasp = Grasp(base_path, file_dict[key])
        A = np.zeros((grasp.total_w.shape[0] * grasp.total_w.shape[1], len(n_m_list)))
        temp_values = []
        
        for idx, (n, m) in enumerate(n_m_list):
            # pri,idx] = temp.flatnt(grasp.expression(n, m).flatten().shape)
            temp = grasp.expression(n, m)
            temp_values.append(temp)
            A[...,idx] = temp.flatten()
                    
        b = grasp.total_w.flatten()
        
        x = np.linalg.inv(A.transpose() @ A) @ A.transpose() @ b 
        
        print(f'distance : {key}', x)
        
        if idx == 0:
            x_list = list(x.copy())
        else:
            x_list.append(list(x)) 
    
        estimation = 0
        for i in range(len(n_m_list)):
            estimation += temp_values[i] * x[i]
        
        grasp.plot_3d(estimation)
        
        plt.show()
    print(x_list)
        
        
    # grasp = Grasp(base_path, file_names)

    # n_m_list = [(0,0), (1,-1), (1,1), (2,0)]
    # # n_m_list = [(0,0), (1,-1), (1,1), (2,-2), (2,0), (2,2)]
    # A = np.zeros((grasp.total_w.shape[0] * grasp.total_w.shape[1], len(n_m_list)))
    
    # temp_values = []
    # for idx, (n, m) in enumerate(n_m_list):
    #     # print(grasp.expression(n, m).flatten().shape)
    #     temp = grasp.expression(n, m)
    #     temp_values.append(temp)
    #     A[...,idx] = temp.flatten()
        
    # b = grasp.total_w.flatten()
    
    # x = np.linalg.inv(A.transpose() @ A) @ A.transpose() @ b
    
    # print('distance : 48.21', x)
    
    # estimation = 0
    # for i in range(len(n_m_list)):
    #     estimation += temp_values[i] * x[i]
    
    # grasp.plot_3d(estimation)
    
    # plt.show()
    
    
    
    
    # estimation = temp_values[0]*z_pistone + temp_values[0]*alpha_h_tilt + temp_values[0]*alpha_v_tilt + temp_values[0]*alpha_oa + z_defocus*alpha_defocus + z_ha*alpha_ha
    
    # z_pistone = grasp.expression(0, 0)
    # alpha_pisonte = grasp.cal_alpha(z_pistone)
    
    # z_h_tilt = grasp.expression(1, 1)
    # alpha_h_tilt = grasp.cal_alpha(z_h_tilt)
    
    # z_v_tilt = grasp.expression(1, -1)
    # alpha_v_tilt = grasp.cal_alpha(z_v_tilt)
    
    # z_oa = grasp.expression(2, -2)
    # alpha_oa = grasp.cal_alpha(z_oa)
    
    # z_defocus = grasp.expression(2, 0)
    # alpha_defocus = grasp.cal_alpha(z_defocus)
    
    # z_ha = grasp.expression(2, 2)
    # alpha_ha = grasp.cal_alpha(z_ha)
    
    # estimation = z_pistone*z_pistone + z_h_tilt*alpha_h_tilt + z_v_tilt*alpha_v_tilt + z_oa*alpha_oa + z_defocus*alpha_defocus + z_ha*alpha_ha
    # estimation = estimation
    
    # print("estimation", estimation)
    # grasp.plot_3d(z_h_tilt*alpha_h_tilt)