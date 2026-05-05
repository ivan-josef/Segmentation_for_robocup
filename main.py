import cv2 
import numpy as np
import pandas as pd 
import pixel_branco as branco
import matplotlib.pyplot as plt
from cv_bridge import CvBridge


class Pixel_Segment:
    def __init__(self,path,lut_csv):
        self.path = path 
        self.lut_csv = pd.read_csv(lut_csv)

    def load_image(self):
        self.img = cv2.imread(self.path)
        self.img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        self.h_img, self.s_img, self.v_img = cv2.split(self.img_hsv)

    def masks(self):
        self.lower = branco.lower
        self.up = branco.up
        self.white_mask = cv2.inRange(self.img_hsv, self.lower, self.up)
        self.green_mask = self.lut_verde[self.h_img,self.s_img,self.v_img]

    def np_lut(self):
        self.lut_verde = np.zeros((180,256,256),dtype=np.uint8)
        self.h_vals = self.lut_csv['H'].values.astype(int)
        self.s_vals = self.lut_csv['S'].values.astype(int)
        self.v_vals = self.lut_csv['V'].values.astype(int)

        self.lut_verde[self.h_vals,self.s_vals,self.v_vals] = 255

    def find_field_boundary_bottom_up(self, mask, tolerancia_pixels=2):
       
        height, width = mask.shape[:2]
        boundary_points = []

        # Percorre cada coluna da imagem de baixo para cima 
        for x in range(width):
            last_green_y = height - 1
            non_green_count = 0
    
            for y in range(height - 1, -1, -1):
                if mask[y, x] > 0:  
                    last_green_y = y
                    non_green_count = 0 
                else:
                    non_green_count += 1
                    if non_green_count >= tolerancia_pixels:
                        break

            # Adiciona o ponto de contorno encontrado
            boundary_points.append([x, last_green_y])
    
        if not boundary_points:
            return None, mask

        boundary_points = np.array(boundary_points, dtype=np.int32) # Converte para inteiro
        hull = cv2.convexHull(boundary_points) # Desenha a envoltoria com base
    
        area_mask = np.zeros_like(mask)
        cv2.fillConvexPoly(area_mask, hull, 255) 

        return hull, area_mask

    def run(self):
        self.load_image()
        self.np_lut()
        self.masks()

        hull, boundary = self.find_field_boundary_bottom_up(self.green_mask)

        img_debug = self.img.copy()
        if hull is not None:
            cv2.polylines(img_debug, [hull], isClosed=True, color=(0, 255, 0), thickness=2)

        # Visualização
        cv2.imshow('Mascara Verde (LUT)', self.green_mask)
        cv2.imshow('Mascara Borda (Area de Jogo)', boundary)
        cv2.imshow('Imagem Original + Convex Hull', img_debug)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

source_lut  = '/home/vtr_caixeta/Segmentation_for_robocup/pixel_verde.csv'
source = '/home/vtr_caixeta/Segmentation_for_robocup/images/bitbots_reality_spl_only_131-16_02_2018__11_18_08_0117_upper_png_jpg_b0.8_s1.0_k0.jpg'
obj = Pixel_Segment(source,source_lut)
obj.run()