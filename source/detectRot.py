# Math 842, Spring 2015
# Detecting molecular symmetry
# Alex Troesch


import numpy as np
import scipy.io
import scipy.ndimage
import scipy.signal
import scipy.stats
import matplotlib.pyplot as plt


def polar_form(img, angular_res=360):
    """ DocString
    """
    
    img_res = img.shape
    x_center = img_res[1]/2
    y_center = img_res[0]/2
    r_max = np.sqrt( np.sum( np.square((x_center, y_center)) ) ).round()
    t_max = angular_res
    output_img = np.zeros((t_max, r_max))
    
    for index in np.ndindex(output_img.shape):
        t = index[0]
        r = index[1]
        x = r*np.cos(2*np.pi*t/t_max) + x_center
        y = r*np.sin(2*np.pi*t/t_max) + y_center
        
        output_img[t,r] = scipy.ndimage.interpolation.map_coordinates(img, [[x],[y]], order=1)
    return output_img


def img_trim(img, epsilon=0):
    """ DocString
    """
    
    img_res = img.shape
    output_img = img
    
    for j in range(img_res[1]):
        if (output_img[:,-1] <= epsilon).all():
            output_img = np.delete(output_img, -1, axis=1)
        else:
            break
            
    return output_img


def img_norm2(img):
    """ DocString
    """
    return auto_cor2(img, 0)


def auto_cor2(img, shift, norm=1):
    """ DocString
    """
    
    shifted_img = np.append(img[shift:], img[:shift], axis=0)
    ac = np.sum(np.vectorize(lambda x,y: x*y)(img, shifted_img))
    
    return ac/norm


def plot_autocor(img, angular_res=360, epsilon=0):
    """ DocString
    """
    
    polar_img = img_trim(polar_form(img, angular_res), epsilon)
    img_norm = img_norm2(polar_img)
    xs = range(polar_img.shape[0])
    ys = [auto_cor2(polar_img, shift, norm=img_norm) for shift in xs]
    
    plt.plot(ys)


def count_rotsym(img, angular_res=360, epsilon=0):
    """ DocString
    """
    
    polar_img = img_trim(polar_form(img, angular_res), epsilon)
    img_norm = img_norm2(polar_img)
    x_max = polar_img.shape[0]
    xs = range(x_max)
    ys = [auto_cor2(polar_img, shift, norm=img_norm) for shift in xs]
    
    # TODO: Better peak estimates
    # Assume no large scale effects over more than 10% of image
    peaks = scipy.signal.find_peaks_cwt(ys, np.arange(x_max/50,x_max/10))
    
    # Always overcount by 1 since, auto_cor2(img,0) is always a maxima
    return len(peaks)-1

# TODO: Move tests to another module
# Load images from the current directory
fileNames = ['1338_c3_top.mat', '5001_c7_top.mat', 'gauss_3_top.mat']

img0 = scipy.io.loadmat(fileNames[0])['images'][0]
img1 = scipy.io.loadmat(fileNames[1])['images'][0]
img2 = scipy.io.loadmat(fileNames[2])['images'][0]

# Count symmetries
print fileNames[0], count_rotsym(img0, 120, epsilon=1) # 3
print fileNames[1], count_rotsym(img1, 120, epsilon=1) # 7
print fileNames[2], count_rotsym(img2, 120, epsilon=1e-01) # 3
