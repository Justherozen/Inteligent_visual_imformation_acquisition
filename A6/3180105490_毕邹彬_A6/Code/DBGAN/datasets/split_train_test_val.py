from pdb import set_trace as st
import os
import numpy as np
import cv2
import argparse
from sklearn.model_selection import train_test_split
import glob
parser = argparse.ArgumentParser('create subdirectories for trainin deblur gan')
parser.add_argument('--fold_blurred', dest='fold_blurred', help='input directory for image A, ex. blurred images', type=str, default='../dataset/50kshoes_edges')
parser.add_argument('--fold_sharp', dest='fold_sharp', help='input directory for image B, ex. sharp images', type=str, default='../dataset/50kshoes_jpg')
parser.add_argument('--output', dest='output', help='output directory', type=str, default='../dataset/test_AB')

args = parser.parse_args()
for arg in vars(args):
    print('[%s] = ' % arg,  getattr(args, arg))
    
folder = args.output
os.makedirs(folder,exist_ok=True)
sharp_out = os.path.join(folder,'sharp')
blurred_out = os.path.join(folder,'blurred')

os.makedirs(folder,exist_ok=True)
os.makedirs(sharp_out,exist_ok=True)
os.makedirs(blurred_out,exist_ok=True)


map_folder={}
for f in [sharp_out,blurred_out]:
    for s in ['train','test','val']:
        map_folder[f+'_'+s]=os.path.join(f,s)
        os.makedirs(map_folder[f+'_'+s],exist_ok=True)
print(map_folder)
#blurred_in = glob.glob(os.path.join(args.fold_A,'*.*g'))
imfiles= glob.glob(os.path.join(args.fold_sharp,'*.*g'))

X_train, X_test, y_train, _= train_test_split(imfiles, np.ones(len(imfiles)), test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.2, random_state=1)


for x in X_train:
    name = x.split('/')[-1]
    dst = os.path.join(map_folder[sharp_out+'_'+'train'],name) 
    
    blur= os.path.join(args.fold_blurred,name)
    if not os.path.exists(blur):
        print('blur image does not exist n/',blur)
        continue
    print('in',x)
    img=cv2.imread(x)
    
    #print(img.shape)
    cv2.imwrite(dst,img)
    print('out',dst)
    #os.symlink(x,dst)
    
    dst_blur = os.path.join(map_folder[blurred_out+'_'+'train'],name)
    img_blur = cv2.imread(blur)
    cv2.imwrite(dst_blur,img_blur)
    #os.symlink(blur,dst_blur)
    
for x in X_test:
    name = x.split('/')[-1]
    dst = os.path.join(map_folder[sharp_out+'_'+'test'],name) 
    
    blur= os.path.join(args.fold_blurred,name)
    if not os.path.exists(blur):
        print('blur image does not exist n/',blur)
        continue
    img=cv2.imread(x)
    print('in',x)
    cv2.imwrite(dst,img)
    print('out',dst)
    #os.symlink(x,dst)
    
    dst_blur = os.path.join(map_folder[blurred_out+'_'+'test'],name)
    img_blur = cv2.imread(blur)
    cv2.imwrite(dst_blur,img_blur)
    #os.symlink(blur,dst_blur)

        
for x in X_val:
    name = x.split('/')[-1]
    dst = os.path.join(map_folder[sharp_out+'_'+'val'],name) 
    
    blur= os.path.join(args.fold_blurred,name)
    if not os.path.exists(blur):
        print('blur image does not exist n/',blur)
        continue
    img=cv2.imread(x)
    print('in',x)
    cv2.imwrite(dst,img)
    print('out',dst)
    #os.symlink(x,dst)
    
    dst_blur = os.path.join(map_folder[blurred_out+'_'+'val'],name)
    img_blur = cv2.imread(blur)
    cv2.imwrite(dst_blur,img_blur)
    #os.symlink(blur,dst_blur)