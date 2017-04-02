import cv2
import numpy as np
from scipy import stats

import cvext as cvx


# PIPELINE
class Options:
    ## define params ##
    # blurring
    
    
    def __init__(self, kernel_size = 3, low_threshold = 60, high_threshold = 160, threshold = 36, rho = 1, theta = 0.05,\
                        min_line_length = 60, max_line_gap = 40, min_angle = 20, max_angle = 70, \
                        line_rotation = 0.5, line_param = 0, line_reps = 0.01, line_aeps = 0.01, fill_mask = 255, min_line_height = 0.6):
        self.kernel_size = kernel_size
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.threshold = threshold
        self.rho = rho
        self.theta = theta
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.line_rotation = line_rotation
        self.line_param = line_param
        self.line_reps = line_reps
        self.line_aeps = line_aeps

        self.fill_mask = fill_mask
        self.min_line_height = min_line_length
    def __call__(self):
        return
    
class Vertex:
    top_left = 0
    top_right = 0
    min_top = 0
    max_top = None
    
    bottom_left = 0
    bottom_right = 0
    
    height = 0
    width = 0
    
    vertices = []
    
    def __init__(self, imshape, min_width1 = 0.45, max_width1 = 0.55, min_width2 = 0.05, max_width2 = 0.95, min_height = 0.55, max_height = None):
        self.width = imshape[1]
        self.height = imshape[0]
        
        self.bottom_left = self.width * min_width2
        self.bottom_right = self.width * max_width2
        
        self.top_left = self.width * min_width1
        self.top_right = self.width * max_width1
        
        self.min_top = self.height * min_height
        if (max_height):
            self.max_top = self.height * max_height
        else: self.max_top = self.height
        
        self.vertices = np.array([[(self.bottom_left, self.max_top), (self.top_left, self.min_top), 
                                   (self.top_right, self.min_top), (self.bottom_right, self.max_top)]], 
                                   dtype=np.int32)
        
        return
    
    def __call__(self):
        return

def get_image(imgpath):
    # read in image file
    imgdata = mpimg.imread(imgpath)
    return imgdata

def get_viewport(image):
    imshape = image.shape
    port = Vertex(imshape)
    return port

def get_options():
    options = Options()
    
    # blending
    options.kernel_size = 3
    # hough lines
    options.rho = 5.5
    options.theta = 1.
    options.threshold = 18
    options.min_len_length = 140
    options.max_line_gap = 60
    # canny edge
    options.low_threshold = 0
    options.high_threshold = 150
    # line processor
    options.line_rotation = 0.42
    options.min_line_height = 0.6
    
    return options

def get_road_options():
    options = Options()
    
    options.kernel_size = 5
    # init params
    options.line_rotation = 0.42
    # hough lines
    options.rho = 3.0
    options.theta = np.pi/180*3.
    options.threshold = 18
    options.min_len_length = 140
    options.max_line_gap = 60
    # canny edge
    options.low_threshold = 120  # 120
    options.high_threshold = 220 # 220
    # line processor
    options.min_line_height = 0.6
    
    return options

### Road Masking ###
# Returns a binary mask with the road edges and noise
def mask_road_image(x, patch_size = [0.8, 0.2], fill_mask = 255):
    patch_region = np.array([[(x.shape[1]*patch_size[1], x.shape[0]*patch_size[0]), (x.shape[1]*patch_size[0], x.shape[0]*patch_size[0]),
                             (x.shape[1]*patch_size[1], x.shape[0]), (x.shape[1]*patch_size[0], x.shape[0])]], dtype=np.int32)
    
    img_blur = cvx.gaussian_blur(x, 9)
    
    imghsl = cvx.to_HLS(img_blur)[:,:,2]
    imgyuv = cvx.to_YUV(img_blur)[:,:,0]
    
    imghsl_patch = cvx.region_of_interest(imghsl, patch_region)
    imgyuv_patch = cvx.region_of_interest(imgyuv, patch_region)
    
    imghsl = np.maximum(0, np.minimum(imghsl - np.max(imghsl_patch), fill_mask))
    imgyuv = np.maximum(0, np.minimum(imgyuv - np.max(imgyuv_patch), fill_mask))
    
    img_mask1 = cvx.color_threshold(imgyuv, thresh=(0,195), mask = fill_mask)
    img_mask2 = cvx.color_threshold(imghsl, thresh=(0,165), mask = fill_mask)
    
    img_mask = cvx.binary_threshold(img_mask1, img_mask2, mask = fill_mask)
    
    img_mask = cvx.gaussian_blur(img_mask, 1)
    
    return img_mask

### Road Edge Detection ###
# Params for detecting road edges
left_road_prev = None
right_road_prev = None

# Finds the road edges
# Returns left and right road edge lines
def find_road_edges(img, options, smooth=True, smooth_delta=0.6):
    global left_road_prev
    global right_road_prev

    width = img.shape[1]
    height = img.shape[0]

    vertices = np.array([[(0, height*0.42), (width, height*0.42), (width, height*0.85), (width/2.,height*0.4), (0, height*0.85)]], dtype=np.int32)

    left_mask_vertices = np.array([[(0, 0),(width/2., 0), (width/2., height), (0, height)]], dtype=np.int32)
    right_mask_vertices = np.array([[(width/2., 0),(width, 0), (width, height), (width/2., height)]], dtype=np.int32)
    
    img_input = mask_road_image(img, fill_mask = options.fill_mask)
    
    canny = cvx.canny(img_input, options.low_threshold, options.high_threshold)
    
    mask = np.zeros_like(canny)
    cv2.fillPoly(mask, vertices, options.fill_mask)
    masked_canny = cv2.bitwise_and(canny, mask)
    
    # left mask
    mask_left = np.zeros_like(canny)
    cv2.fillPoly(mask_left, left_mask_vertices, options.fill_mask)
    masked_canny_left = cv2.bitwise_and(masked_canny, mask_left)
    
    #left line
    left_points = cv2.findNonZero(masked_canny_left)
    
    [lm, lc, lr, lp, ls] = stats.linregress(left_points[:,0][:,0], left_points[:,0][:,1])
    [lx0, ly0, lx1, ly1] = [0, cvx.y_intercept(0, lm, lc), width/2., cvx.y_intercept(width/2, lm, lc)]
    
    # right mask
    mask_right = np.zeros_like(canny)
    cv2.fillPoly(mask_right, right_mask_vertices, options.fill_mask)
    masked_canny_right = cv2.bitwise_and(masked_canny, mask_right)
    
    #right line
    right_points = cv2.findNonZero(masked_canny_right)
    [rm, rc, rr, rp, rs] = stats.linregress(right_points[:,0][:,0], right_points[:,0][:,1])
    [rx0, ry0, rx1, ry1] = [width/2, cvx.y_intercept(width/2, rm, rc), width, cvx.y_intercept(width, rm, rc)]
    
    # get intersection
    ixy = cvx.line_intercept((lx0, ly0, lx1, ly1), (rx0, ry0, rx1, ry1))
    ixy = ((lx1+rx0+ixy[0]) / 3., (ly1+ry0+ixy[1]) / 3.)
    
    diff_y = (((ly1+ry0)/2.)-ixy[1])
    
    left = np.array([int(lx0), int(ly0+diff_y), int(ixy[0]), int(ixy[1])])
    right = np.array([int(ixy[0]), int(ixy[1]), int(rx1), int(ry1+diff_y)])
    
    if (left_road_prev is not None and smooth):
        left = np.average((left, left_road_prev), weights=(smooth_delta, 1.0-smooth_delta), axis=0)
        left_road_prev = np.copy(left)
    
    if (right_road_prev is not None and smooth):
        right = np.average((right, right_road_prev), weights=(smooth_delta, 1.0-smooth_delta), axis=0)
        right_r_prev = np.copy(right)
    
    left = left.astype(int)
    right = right.astype(int)
    
    return [left, right]

### Finds Road Edge Distances ###
# Finds the road edge distance lines
# Returns: list of distance lines from both sides and corresponding road edge lines.
def find_edge_distances(image, options, x_coords=[0.05, 0.1, 0.2, 0.8, 0.9, 0.95], \
                       y_coords=[0.6, 0.5, 0.4, 0.4, 0.5, 0.6], centroid = None):
    [left, right] = find_road_edges(image, options)
    
    left_lines = []
    right_lines = []
    
    roadx = cvx.line_intercept(left, right)
    
    if (centroid == None):
        centroid = (image.shape[1]/2., image.shape[0])
    
    for x in range(len(x_coords)):
        # get positions
        x_pos = x_coords[x] * image.shape[1]
        y_pos = y_coords[x] * image.shape[0]
        
        # scan line
        line = [x_pos, y_pos, centroid[0], centroid[1]]
        
        ixy = None
        # check horizon (left or right)
        ixy_l = cvx.line_intercept(line, left)
        ixy_r = cvx.line_intercept(line, right)
        
        # get closest intercept
        if ((ixy_l != None) and (ixy_r != None)):
            if (ixy_l[1] >= ixy_r[1]):
                ixy = ixy_l
            else:
                ixy = ixy_r
        
        # if intersection, shorten line
        if (ixy != None):
            line = [ixy[0], ixy[1], centroid[0], centroid[1]]
        
        if (x_pos <= roadx[0]):
            left_lines.append(line)
        else:
            right_lines.append(line)
    
    # compute left distances
    left_dx = []
    for [x0, y0, x1, y1] in left_lines:
        left_dx.append(cvx.distance(x0, y0, x1, y1))
    # compute right distances
    right_dx = []
    for [x0, y0, x1, y1] in right_lines:
        right_dx.append(cvx.distance(x0, y0, x1, y1))

    left_dx = np.array(left_dx)
    right_dx = np.array(right_dx)
    
    return [left_dx, right_dx]

def preprocess(img, options, viewport):
    yuv = cvx.to_HSV(image)
    yuv_mask = cvx.color_threshold(yuv, thresh=(options.low_threshold, options.high_threshold), channel=2)
    hls = cvx.to_HLS(image)
    hls_mask = cvx.color_threshold(hls, thresh=(options.low_threshold, options.high_threshold), channel=2)
    mask = cvx.binary_threshold(yuv_mask, hls_mask)
    img_proc = cv2.bitwise_and(image, image, mask)
    
    img_proc = cvx.to_GRAY(img_proc)
    
    img_proc = cvx.gaussian_blur(img_proc, options.kernel_size)
    
    img_proc = cvx.canny(img_proc, options.low_threshold, options.high_threshold)
    
    img_proc = cvx.region_of_interest(img_proc, viewport.vertices)
    
    return img_proc

def process(imgdata, options, viewport):
    
    imggray = preprocess(imgdata, options, viewport)
    
    lines = cvx.hough_lines(imggray, options.rho, options.theta, options.threshold,
                        options.min_line_length, options.max_line_gap)
    
    lines = cvx.process_lines(lines, imggray.shape[1], imggray.shape[0], viewport, options.line_rotation, 
                          options.line_param, options.line_reps, options.line_aeps)
    line_img = np.zeros((*imggray.shape, 3), dtype=np.uint8)
    cvx.draw_lines(line_img, lines)
    
    imgresult = cvx.weighted_img(line_img, imgdata)
    
    return imgresult
