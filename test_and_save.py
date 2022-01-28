import numpy as np
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import glob, os, re
import scipy.io
import pickle
from MODEL import model
import time
from scipy.io import savemat
import argparse
import math
import copy
from skimage.metrics import structural_similarity as ssim


parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
model_path = args.model_path

DATA_PATH = "./data/test/"
save_dir  = "./data/results/"


def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    
    if mse > 1000:
        return -100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    
def calculate_psnr_ssim(out,gt):
    crop = 22
    H,W = gt.shape
    gt_img = np.zeros((H//7, W//7, 7, 7))
    out_img = np.zeros((H//7, W//7, 7, 7))
    for ax in range(7):
        for ay in range(7):
            gt_img[:, :, ay, ax] = gt[ay::7, ax::7]
            out_img[:, :, ay, ax] = out[ay::7, ax::7]
            
    psnr_total = np.zeros((7, 7))
    ssim_total = np.zeros((7, 7))
    for ax in range(7):
        for ay in range(7):
            psnr_total[ay,ax] = psnr(out_img[crop:-crop, crop:-crop, ay, ax] , gt_img[crop:-crop, crop:-crop, ay, ax]) 
            ssim_total[ay,ax] = ssim((out_img[crop:-crop, crop:-crop, ay, ax] * 255.0).astype(np.uint8), (gt_img[crop:-crop, crop:-crop, ay, ax] * 255.0).astype(np.uint8),gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
            
    psnr_total[0,0] = 0    
    psnr_total[3,0] = 0 
    psnr_total[6,0] = 0
    psnr_total[0,3] = 0    
    psnr_total[3,3] = 0 
    psnr_total[6,3] = 0
    psnr_total[0,6] = 0    
    psnr_total[3,6] = 0 
    psnr_total[6,6] = 0
    
    ssim_total[0,0] = 0    
    ssim_total[3,0] = 0 
    ssim_total[6,0] = 0
    ssim_total[0,3] = 0    
    ssim_total[3,3] = 0 
    ssim_total[6,3] = 0
    ssim_total[0,6] = 0    
    ssim_total[3,6] = 0 
    ssim_total[6,6] = 0

    average_psnr = np.sum(psnr_total)/40 
    average_ssim = np.sum(ssim_total)/40
    
    return average_psnr, average_ssim
    
def ycbcr2rgb(ycbcr):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = copy.deepcopy(ycbcr)
    rgb[:,0] -= 16. / 255.
    rgb[:,1:] -= 128. / 255.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    return rgb.clip(0, 1).reshape(shape).astype(np.float32)
    
    
def save_img(out,directory):
    out_img = ycbcr2rgb(out)
    H,W,D = out_img.shape
    h = H//7
    w = W//7
    img = np.zeros((h,w,D))
    for ax in range(7):
        for ay in range(7):
            img = out_img[ay::7,ax::7,:]
            img = cv2.cvtColor(img * 255, cv2.COLOR_BGR2RGB)
            img_name = '/{}_{}.png'.format(ay+1, ax+1)
            save_path = directory + img_name
            cv2.imwrite(save_path , img)

def test_with_sess(epoch, ckpt_path, data_path,sess):
    saver.restore(sess, tf.train.latest_checkpoint('C:/Ahmed/LF_Raw/checkpoints/'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    psnr_dict = {}
    folder_list = glob.glob(os.path.join(data_path,"*"))
    print("folder_list",folder_list)
    for folder_path in folder_list:
        print("folder_path:  ", folder_path)
        folder_name = os.path.basename(folder_path)
        folder_dir = os.path.join(save_dir,folder_name)
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)
        psnr_list = []
        img_gt_list = glob.glob(os.path.join(folder_path,'gt',"*"))
        img_in_list = glob.glob(os.path.join(folder_path,'in',"*"))
        average_psnr = 0
        average_ssim = 0
        average_time = 0
        for i in range(len(img_gt_list)):
            img_path = img_gt_list[i]
            img_name = os.path.basename(img_path)[:-4]
            img_dir = os.path.join(save_dir,folder_name,img_name)
            print("img_path:  ", img_path)
            print("img_name:  ", img_name)
            print("img_dir:  ", img_dir)
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
                
            gt_img = scipy.io.loadmat(img_path)['gt']
            input_img = scipy.io.loadmat(img_in_list[i])['in']
            print("gt_img: ", gt_img.shape)
            print("input_img: ", input_img.shape)
            input_y = input_img[:,:,0]
            gt_y = gt_img[:,:,0]
            start_t = time.time()
            img_out_y = sess.run([output_tensor], feed_dict={input_tensor: np.resize(input_y, (1, input_y.shape[0], input_y.shape[1], 1))})
            img_out_y = np.resize(img_out_y, (gt_y.shape[0], gt_y.shape[1]))
            end_t = time.time()
            psnr_value, ssim_value = calculate_psnr_ssim(img_out_y,gt_y)
            print("PSNR and SSIM:  ", psnr_value, ssim_value)
            average_psnr += psnr_value
            average_ssim += ssim_value
            output_img = input_img
            output_img[:,:,0] = img_out_y
            
            print("end_t",end_t,"start_t",start_t)
            reconstruction_time = (end_t-start_t)/49
            average_time += reconstruction_time
            print("average time consumption",reconstruction_time)
            save_img(output_img,img_dir)
            
            with open(img_dir+'/result.txt', 'w') as f:
                Metric_values = '{} scene: Average_PSNR: {:.2f}, Average_SSIM: {:.3f}, Average reconstruction time: {:.3f}'.format(img_name, psnr_value, ssim_value, reconstruction_time)
                f.write(Metric_values)

        average_psnr = average_psnr/len(img_gt_list)  
        print("average_ssim:  ", average_ssim)        
        average_ssim = average_ssim/len(img_gt_list)
        print("average_ssim:  ", average_ssim)
        print("len(img_gt_list):  ", len(img_gt_list))
        average_time = average_ssim/len(img_gt_list)
        with open(folder_dir+'/result.txt', 'w') as f:
            Metric_values = '{} dataset: Average_PSNR: {:.2f}, Average_SSIM: {:.3f}, Total Time: {}, Average reconstruction time: {:.3f}'.format(folder_name, average_psnr, average_ssim, reconstruction_time*49*len(img_gt_list),reconstruction_time)
            f.write(Metric_values)        

           
            


            
        psnr_dict[os.path.basename(folder_path)] = psnr_list
    with open('psnr/%s' % os.path.basename(ckpt_path), 'wb') as f:
        pickle.dump(psnr_dict, f)
def test_VDSR(epoch, ckpt_path, data_path):
	with tf.Session() as sess:
		test_with_sess(epoch, ckpt_path, data_path, sess)
if __name__ == '__main__':
    model_list = sorted(glob.glob("./checkpoints/LF_epoch_*"))
    model_list = [fn for fn in model_list if not os.path.basename(fn).endswith("meta")]
    model_list = [fn for fn in model_list if not os.path.basename(fn).endswith("index")]
    with tf.Session() as sess:
        input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
        shared_model = tf.make_template('shared_model', model)
        output_tensor, weights 	= shared_model(input_tensor)
        saver = tf.compat.v1.train.Saver(weights)
        tf.initialize_all_variables().run()
        for model_ckpt in model_list:
            epoch = int(model_ckpt.split('epoch_')[-1].split('.ckpt')[0])
            test_with_sess(epoch, model_ckpt, DATA_PATH,sess)
