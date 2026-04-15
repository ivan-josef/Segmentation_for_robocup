import cv2 
import numpy as np
import pandas as pd 
import def_white_threshold as branco

class Pixel_Segment:
    def __init__(self,path,lut_csv):
        self.path = path 
        self.lut_csv = pd.read_csv(lut_csv)

        self.load_image()
        self.grid_height = int(self.height / 8)
        self.grid_widht = int(self.width / 8)

    def load_image(self):
        self.img = cv2.imread(self.path)
        self.img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.h_img, self.s_img, self.v_img = cv2.split(self.img_hsv)
        self.height, self.width, self.channels = self.img.shape

    def masks_and_resize(self):
        self.lower = branco.lower
        self.up = branco.up
        self.white_mask = cv2.inRange(self.img_hsv, self.lower, self.up)
        self.green_mask = self.lut_verde[self.h_img,self.s_img,self.v_img]

        self.white_mask_resized = cv2.resize(self.white_mask,(self.grid_height,self.grid_widht))
        self.green_mask_resized = cv2.resize(self.green_mask,(self.grid_height,self.grid_widht))


    def np_lut(self):
        self.lut_verde = np.zeros((180,256,256),dtype=np.uint8)
        self.h_vals = self.lut_csv['H'].values.astype(int)
        self.s_vals = self.lut_csv['S'].values.astype(int)
        self.v_vals = self.lut_csv['V'].values.astype(int)

        self.lut_verde[self.h_vals,self.s_vals,self.v_vals] = 255

    def run(self):
        self.load_image()
        self.np_lut()
        self.masks_and_resize()
        cv2.imshow('imagem',self.img)
        print(self.height,self.width)
        cv2.imshow('white_mask',self.white_mask_resized)
        cv2.imshow('green_mask',self.green_mask_resized)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        


source_lut  = 'green_pixels.csv'
source = 'images/bitbots_reality_spl_only_131-16_02_2018__11_18_08_0117_upper_png_jpg_b0.8_s1.0_k0.jpg'
obj = Pixel_Segment(source,source_lut)
obj.run()