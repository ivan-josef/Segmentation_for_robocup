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
        self.img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.h_img, self.s_img, self.v_img = cv2.split(self.img_hsv)
        self.height, self.width, self.channels = self.img.shape

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
        uniq_img = uniq_img_cal.astype(np.uint8) 

        # soma de valores da mesma linha 
        soma_por_linha = np.sum(uniq_img, axis=1)
        row_sums_1d = soma_por_linha + np.roll(soma_por_linha, 1)
        row_sums_1d[0] = soma_por_linha[0]  
        matriz_row_sum = np.broadcast_to(row_sums_1d.reshape(self.grid_height, 1), (self.grid_height, self.grid_width))

        # soma da janela
        kernel_8x4 = np.ones((4, 8), dtype=np.float32)
        matriz_window_sum = cv2.filter2D(uniq_img.astype(np.float32), -1, kernel_8x4, anchor=(4, 0))  

        #operação de binarização
        passou_no_pix  = uniq_img >= self.tpix
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
        print(self.histogram)
        

        #filtragem e convexhull
        y_vals = np.array([item[0] for item in self.histogram], dtype=np.float32)
        y_median = medfilt(y_vals, kernel_size=3)
        y_gauss = gaussian_filter1d(y_median, sigma=1.0)
        contour_points = []


        for x in range(self.grid_width):
            suavized_y = int(np.clip(y_gauss[x], 0, self.grid_height - 1))
            contour_points.append([x, suavized_y])

        # Adicionar os cantos inferiores para fechar o polígono do campo
        contour_points.append([self.grid_width - 1, self.grid_height - 1]) 
        contour_points.append([0, self.grid_height - 1])                   

        # Converter para o formato (N, 1, 2) int32 do OpenCV
        pontos_np = np.array(contour_points, dtype=np.int32).reshape((-1, 1, 2))

        # Calcular o Casco Convexo sobre a borda suavizada
        hull = cv2.convexHull(pontos_np)

        # Criar a máscara convexa final
        self.mascara_convexa = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        cv2.drawContours(self.mascara_convexa, [hull], -1, 255, thickness=cv2.FILLED)
            

        self.mascara_tamanho_original = cv2.resize(
            self.mascara_convexa, 
            (self.width, self.height), 
            interpolation=cv2.INTER_NEAREST
        ) 
        self.visao_recortada = cv2.bitwise_and(self.img, self.img, mask=self.mascara_tamanho_original)
         

    def run(self):
        self.masks_and_resize()
        self.binarization()

        cv2.imshow('imagem',self.img)
        cv2.imshow('white_mask',self.visao_recortada)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        


source_lut  = 'green_pixels.csv'
source = 'images/bitbots_reality_spl_only_131-16_02_2018__11_18_08_0117_upper_png_jpg_b0.8_s1.0_k0.jpg'
obj = Pixel_Segment(source,source_lut)
obj.run()