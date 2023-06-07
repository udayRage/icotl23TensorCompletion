import numpy as np
import pandas as pd
import tensorly as tl
import cv2

def get_x_point(point):
    return float(point[6:-1].split()[0])

def get_y_point(point):
    return float(point[6:-1].split()[1])

def df2np(df, separator = '\t'):

    # change header param if it exists
    # df = pd.read_csv(df_file, sep=separator, header=None)
    points = df[df.columns[0]]

    x_points = points.apply(get_x_point).unique()
    y_points = points.apply(get_y_point).unique()

    num_bands = len(df.columns) - 1
    x_image_map = {}
    y_image_map = {}

    for i in range(len(x_points)):
        x_image_map[x_points[i]] = i

    for i in range(len(y_points)):
        y_image_map[y_points[i]] = i


    image_frame = np.zeros((len(x_points), len(y_points),num_bands))
    df = df[df.columns[1:]]
    image_vec = df.to_numpy(dtype = 'float64')

    for i in range(len(image_vec)):
        for band_num in range(num_bands):
            image_frame[x_image_map[get_x_point(points[i])], y_image_map[get_y_point(points[i])],band_num] = image_vec[i][band_num]

    gap_x = x_points[1] - x_points[0]
    gap_y = y_points[1] - y_points[0]

    starting_x = x_points[0] - (gap_x/2)
    starting_y = y_points[0] - (gap_y/2)
        
    geo_coordinates = (starting_x, gap_x, 0.0, starting_y, 0.0, gap_y)

    return image_frame, points


def mask_on_img(img, mask):
    '''
        def: apply 2D mask on 3D img
    '''
    for i in range(img.shape[-1]):
        img[:,:,i] *= mask
    
    return img

def masker(img, threshold = 0.1):

    imSize = img.shape
    if threshold == 0.1:
        threshold = np.sum(img[:,:,0])/(imSize[0]*imSize[1])
        print("Auto threshold: ", threshold)
    
    # np.sum(img[:,:,0])/(imSize[0]*imSize[1])
    mask = np.zeros(shape = imSize[:-1])
    for r in range(1,imSize[0]-1):
        for c in range(1,imSize[1]-1):
            nbrs = np.sum(img[r+1,c,:]) + np.sum(img[r-1,c,:])
            # +img[r,c+1,:]+img[r,c-1,:]
            nbrs /= 2
            if abs(np.sum(img[r,c,:]) - nbrs) > threshold or np.sum(img[r,c,:]) < 1e-4:
                mask[r,c] = 1

    # border check
    for r in [0,imSize[0]-1]:
        for c in [0,imSize[1]-1]:
            if np.sum(img[r,c,:]) < 1e-4:
                mask[r,c] = 1
    mask = np.array(mask,dtype=bool)
    mask = ~mask

    return mask

def exact_masker(img):

    imSize = img.shape
    mask = np.ones(shape = imSize[:-1])

    for r in range(0,imSize[0]-1):
        for c in range(0,imSize[1]-1):
            if np.sum(img[r,c,:]) < 1e-4:
                mask[r,c] = 0
    mask = np.array(mask,dtype=bool)

    return mask

def copyMask(img_src, img_dest, mask):
    '''
        Copy mask from img_src to img_dest
    '''
    for i in range(img_src.shape[0]):
        for j in range(img_src.shape[1]):
            if mask[i][j] == 1:
                img_dest[i][j] = img_src[i][j]
    return img_dest
    
def maskCheck(img_corrupt, mask):
    for i in range(img_corrupt.shape[0]):
        for j in range(img_corrupt.shape[1]):
            if mask[i][j] != (np.sum(img_corrupt[i][j]) > 0):
                print(mask[i][j])
                print(np.sum(img_corrupt[i][j]))
                return False
    return True

def normalize(img, mask):
    for i in range(img.shape[-1]):
        layer = img[:,:,i]
        layer_avg = np.sum(layer*mask) + 0.000000001
        layer_avg /= len(np.nonzero(mask)[0]) # no. of non-zero elements

        new_avg = np.sum(layer*~mask) + 0.000000001
        new_avg /= len(np.nonzero(~mask)[0])

        img[:,:,i] = layer * mask + layer * ~mask * (layer_avg/new_avg)
    
    return img

def replaceRows(X, img, perc=0.5):
    imSize = img.shape
    # only x
    mask = np.zeros(imSize[:-1], dtype=bool)
    for i in range(len(imSize[0])):
        if np.random.rand() < perc:
            X[i, :, :] = img[i, :, :]
            mask[i, :] = 1
    return X, mask

def replacePixels(X, known, img):
    imSize = img.shape
    # only x,y
    mask = np.zeros(imSize[:-1], dtype=bool)
    for i in range(len(known)):
        in1 = int(np.ceil(known[i] / imSize[1]) - 1) # x coordinate
        in2 = int(known[i] % imSize[1]) # y coordinate
        X[in1, in2, :] = img[in1, in2, :]
        mask[in1, in2] = 1
    return X, mask

# helper
def replaceLayer(X, known, img, layer = 0):
    imSize = img.shape
    # only x,y
    mask = np.zeros(imSize[:-1], dtype=bool)
    for i in range(len(known)):
        in1 = int(np.ceil(known[i] / imSize[1]) - 1) # x coordinate
        in2 = int(known[i] % imSize[1]) # y coordinate
        X[in1, in2, layer] = img[in1, in2, layer]
        mask[in1, in2] = 1
    return X, mask

def randomDelete(img, perc=0.5, layer = -1):
    '''
        @param layer: Delete pixels from this layer. -1 for all layers
        Randomly remove pixels/layer data and keep only `perc` fraction of entries
    '''
    imSize = img.shape
    known = np.arange(np.prod(imSize) / imSize[2])
    np.random.shuffle(known)
    known = known[:int(perc * (np.prod(imSize) / imSize[2]))] # keep only known idx
    # print(known.shape)

    X = np.zeros(imSize)
    if layer == -1:
        X, msk = replacePixels(X, known, img)
    else:
        X, msk = replaceLayer(X, known, img, layer)
        X[~msk] = img[~msk]
    return X, msk

def shrinkage(X, t):
    '''
        Shrinkage operator
    '''
    U, Sig, VT = np.linalg.svd(X,full_matrices=False)

    Temp = np.zeros((U.shape[1], VT.shape[0]))
    for i in range(len(Sig)):
        Temp[i, i] = Sig[i]  
    Sig = Temp

    Sigt = Sig
    imSize = Sigt.shape

    for i in range(imSize[0]):
        Sigt[i, i] = np.max(Sigt[i, i] - t, 0)

    temp = np.dot(U, Sigt)
    T = np.dot(temp, VT)
    return T

def shrinkageTop(X, t, top):
    '''
        Shrinkage operator
    '''
    U, Sig, VT = np.linalg.svd(X,full_matrices=False)

    Temp = np.zeros((U.shape[1], VT.shape[0]))
    for i in range(len(Sig)):
        Temp[i, i] = Sig[i]  
    Sig = Temp

    Sigt = Sig
    imSize = Sigt.shape

    for i in range(imSize[0]):
        Sigt[i, i] = np.max(Sigt[i, i] - t, 0)
    
    for i in range(top):
        # removing top values for mask since majority is black
        Sigt[i,i] = 0

    temp = np.dot(U, Sigt)
    T = np.dot(temp, VT)
    return T

def Trimming(X, t):
    '''
        Trimming operator
    '''
    U, Sig, VT = np.linalg.svd(X,full_matrices=False)

    Temp = np.zeros((U.shape[1], VT.shape[0]))
    for i in range(len(Sig)):
        Temp[i, i] = Sig[i]  
    Sig = Temp

    Sigt = Sig
    imSize = Sigt.shape

    for i in range(imSize[0]):
        Sigt[i, i] = np.minimum(Sigt[i, i], t)

    temp = np.dot(U, Sigt)
    T = np.dot(temp, VT)
    return Sig, T

def RSEtiff(gt, fixed):

    '''
        Get RSE. arg order is: ground truth, fixed image.
    '''
    return tl.norm((fixed-gt)*255,2)/tl.norm(gt*255,2)

def RSE(gt, fixed):

    '''
        Get Relative Squared Error. arg order is: ground truth, fixed image.
    '''
    return tl.norm(fixed-gt,2)/tl.norm(gt,2)

def MSE(gt, fixed):

    '''
        Get Mean Squared Error. arg order is: ground truth, fixed image.
    '''
    return tl.norm(fixed-gt,2)

def show(image, caption = "Image"):
    cv2.namedWindow(caption, cv2.WINDOW_NORMAL)
    cv2.imshow(caption, image.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()