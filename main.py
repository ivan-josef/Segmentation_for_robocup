import cv2 
import numpy as np
import pandas as pd 
import def_white_threshold as branco

from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

class Pixel_Segment:
    def __init__(self,path,lut_csv):
        self.path = path 
        self.lut_csv = pd.read_csv(lut_csv)

        self.load_image()
        self.grid_height = int(self.height / 8)
        self.grid_width = int(self.width / 8)

        self.tpix = 20
        self.trow = 10000
        self.twin = 3000

        self.histogram = []

    def load_image(self):
        self.img = cv2.imread(self.path)
        self.height, self.width, self.channels = self.img.shape
        self.img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.h_img, self.s_img, self.v_img = cv2.split(self.img_hsv)

    def masks_and_resize(self):
        #white mask
        lower = branco.lower
        up = branco.up

        self.white_mask = cv2.inRange(self.img_hsv, lower, up)

        #green mask
        self.lut_verde = np.zeros((180,256,256),dtype=np.uint8)
        h_vals = self.lut_csv['H'].values.astype(int)
        s_vals = self.lut_csv['S'].values.astype(int)
        v_vals = self.lut_csv['V'].values.astype(int)

        self.lut_verde[h_vals,s_vals,v_vals] = 255
        self.green_mask = self.lut_verde[self.h_img,self.s_img,self.v_img]

        #masks resize
        self.white_mask_resized = cv2.resize(self.white_mask,(self.grid_width,self.grid_height),interpolation=cv2.INTER_AREA).astype(np.float32)
        self.green_mask_resized = cv2.resize(self.green_mask,(self.grid_width,self.grid_height),interpolation=cv2.INTER_AREA).astype(np.float32)

    def binarization(self):
        #fundindo as imagens de cor 
        uniq_img_cal = (self.white_mask_resized * 0.5) + (self.green_mask_resized * 1.0)
        uniq_img_cal = np.clip(uniq_img_cal,0,255)
        self.uniq_img = uniq_img_cal.astype(np.uint8) 

        # soma de valores da mesma linha 
        soma_por_linha = np.sum(self.uniq_img, axis=1)
        row_sums_1d = soma_por_linha + np.roll(soma_por_linha, 1)
        row_sums_1d[0] = soma_por_linha[0]  
        matriz_row_sum = np.broadcast_to(row_sums_1d.reshape(self.grid_height, 1), (self.grid_height, self.grid_width))

        # soma da janela
        kernel_8x4 = np.ones((4, 8), dtype=np.float32)
        matriz_window_sum = cv2.filter2D(self.uniq_img.astype(np.float32), -1, kernel_8x4, anchor=(4, 0))  

        #operação de binarização
        passou_no_pix  = self.uniq_img >= self.tpix
        passou_no_row = matriz_row_sum >= self.trow
        passou_na_win = matriz_window_sum >= self.twin

        self.borda_binaria = (passou_no_pix & (passou_no_row | passou_na_win)).astype(np.uint8) * 255
        
        # histograma
        lines = len(self.borda_binaria)
        colums = len(self.borda_binaria[0])


        for col in range(colums):
            zero_count = 0
            last_one = lines -1
            for line in range(lines -1,-1,-1):
                if  self.borda_binaria[line][col] == 255:
                    zero_count = 0
                    last_one = line 
                else:
                    zero_count+=1
                if zero_count >= 4:
                    break
            self.histogram.append([last_one,col])
        

        #filtragem e convexhull
        y_vals = np.array([item[0] for item in self.histogram], dtype=np.float32)
        y_median = medfilt(y_vals, kernel_size=3)
        self.y_gauss = gaussian_filter1d(y_median, sigma=1.0)
        contour_points = []


        for x in range(self.grid_width):
            suavized_y = int(np.clip(self.y_gauss[x], 0, self.grid_height - 1))
            contour_points.append([x, suavized_y])

        contour_points.append([self.grid_width - 1, self.grid_height - 1]) 
        contour_points.append([0, self.grid_height - 1])                   

        pontos_np = np.array(contour_points, dtype=np.int32).reshape((-1, 1, 2))

        self.hull = cv2.convexHull(pontos_np)

        self.mascara_convexa = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        cv2.drawContours(self.mascara_convexa, [self.hull], -1, 255, thickness=cv2.FILLED)
            

    
    def skeletonization_and_connect(self):


        #skeletização

        self.mascara_tamanho_original = cv2.resize(
            self.mascara_convexa, 
            (self.width, self.height), 
            interpolation=cv2.INTER_NEAREST
        ) 
        
        self.field_line_mask = cv2.bitwise_and(self.white_mask,self.mascara_tamanho_original)
        self.mascara_relevo = cv2.GaussianBlur(self.field_line_mask,(31,31),0)


        
        vizinho_cima = np.roll( self.mascara_relevo,1,axis=0)
        passou_cima = vizinho_cima >= self.mascara_relevo
        vizinho_baixo = np.roll(self.mascara_relevo,-1,axis=0)
        passou_baixo = vizinho_baixo >= self.mascara_relevo
        vizinho_direita = np.roll(self.mascara_relevo,-1,axis=1)
        passou_direita = vizinho_direita >= self.mascara_relevo
        vizinho_esquerda = np.roll(self.mascara_relevo,1,axis=1)
        passou_esquerda = vizinho_esquerda >= self.mascara_relevo
        vizinho_dig_sup_dir = np.roll(self.mascara_relevo,(1,-1),axis=(0,1))
        passou_dig_sup_dir = vizinho_dig_sup_dir >= self.mascara_relevo
        vizinho_dig_inf_dir = np.roll(self.mascara_relevo,(-1,-1),axis=(0,1))
        passou_dig_inf_dir = vizinho_dig_inf_dir >= self.mascara_relevo
        vizinho_dig_sup_esq = np.roll(self.mascara_relevo,(1,1),axis=(0,1))
        passou_dig_sup_esq = vizinho_dig_sup_esq >= self.mascara_relevo
        viziho_dig_inf_esq = np.roll(self.mascara_relevo,(-1,1),axis=(0,1))
        passou_dig_inf_esq = viziho_dig_inf_esq >= self.mascara_relevo
        
        c_xy = np.sum([
            passou_cima, 
            passou_baixo, 
            passou_direita, 
            passou_esquerda, 
            passou_dig_sup_dir, 
            passou_dig_inf_dir, 
            passou_dig_sup_esq, 
            passou_dig_inf_esq
        ], axis=0, dtype=np.uint8)

        nao_e_fundo = self.mascara_relevo > 0
        e_cume_montanha = c_xy < 3 
        skeleton = nao_e_fundo & e_cume_montanha
        self.skeleton_img = np.where(skeleton,255,0).astype(np.uint8)

    
        


    def debug(self):
        self.masks_and_resize()
        self.binarization()
        self.skeletonization_and_connect()

        img_copy = self.img.copy()

        dict_debug = {}
        dict_debug['white_mask'] = self.white_mask
        dict_debug['green_mask'] = self.green_mask
        dict_debug['white_mask_resized'] = self.white_mask_resized
        dict_debug['green_mask_resized'] = self.green_mask_resized
        dict_debug['uniq_img_masks'] = self.uniq_img
        dict_debug['borda_binaria'] = self.borda_binaria

        raw_points = []
        for x in range(self.grid_width):
            real_x = x*8
            real_y = int(self.y_gauss[x]*8)
            raw_points.append([real_x,real_y])

        pre_hull_points = np.array(raw_points,dtype=np.int32).reshape((-1,1,2))
        cv2.polylines(img_copy,[pre_hull_points],isClosed=False,color=(0,0,255),thickness=2)

        hull_points = self.hull * 8
        cv2.polylines(img_copy,[hull_points], isClosed=True, color=(0, 255, 255), thickness=2)

        dict_debug['img_edge'] = img_copy
        dict_debug['lines_relevo'] = self.mascara_relevo
        dict_debug['skeleton'] = self.skeleton_img

        keys = list(dict_debug.keys())
        index = 0

        while True:
            key = keys[index]
            value = dict_debug[key]
            if value.shape[0] != self.height or value.shape[1] != self.width:
                img_resized = cv2.resize(
                value,
                (self.width, self.height),
                interpolation=cv2.INTER_NEAREST
                )
            else:
                img_resized = value.copy()

            cv2.putText(img_resized, key, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255,255,255), 2)

            cv2.imshow("debug", img_resized)
            k = cv2.waitKey(0) & 0xFF

            if k == ord('d'):  
                index = (index + 1) % len(keys)

            elif k == ord('a'):  
                index = (index - 1) % len(keys)

            elif k == ord('q'):  
                break

        cv2.destroyAllWindows()

        


    def run(self):
        self.masks_and_resize()
        self.binarization()
        self.skeletonization_and_connect()

        cv2.imshow('mascara maciça',self.mascara_relevo)
        cv2.imshow('resultado da janela 3x3',self.skeleton_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        


source_lut  = 'green_pixels.csv'
source = 'images/image_2.png'
obj = Pixel_Segment(source,source_lut)
obj.debug()