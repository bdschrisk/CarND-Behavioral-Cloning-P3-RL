import math
from math import atan2, degrees, pi
from scipy import stats
import numpy as np
import cv2

def to_GRAY(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def to_HSV(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

def to_HLS(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

def to_YUV(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

def threshold(img, minimum = 200, maximum = 255):
    return cv2.threshold(img, minimum, maximum, cv2.THRESH_BINARY)

def color_threshold(img, thresh=(0, 255), channel = 2, mask = 1):
    c_data = None
    
    if len(img.shape) > 2 and img.shape[2] > channel:
        c_data = img[:,:,channel]
    else:
        c_data = img[:,:]
    
    c_bin = np.zeros_like(c_data)
    
    c_bin[(c_data >= thresh[0]) & (c_data <= thresh[1])] = mask
    return c_bin

def binary_threshold(mask1, mask2, mask = 1):
    output = np.zeros_like(mask1)
    output[(mask1 == mask) & (mask2 == mask)] = mask
    return output

def image_mask(img, mask):
    return cv2.bitwise_and(img, mask)

def contours(threshold):
    contours = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]

def fitLine(contour, param = 0, reps = 0, aeps = 0.01):
    return cv2.fitLine(contour, cv2.DIST_L2, param, reps, aeps)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def angle(x0, y0, x1, y1):
    # compute deltas
    dx = x1 - x0
    dy = y1 - y0
    # angle and conversion
    rads = atan2(dy, dx)
    
    return degrees(rads)

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def y_intercept(x, m, c):
    return (m * x) + c

def x_intercept(y, m, c):
    return (y - c) / m

def not_number(val1):
    return (math.isnan(val1) or math.isinf(val1))

# computes the interception point of two lines
def line_intercept(line1, line2, epsilon = 1e-08):
    a1 = (line1[1] - line1[3]) / (line1[0] - line1[2])
    b1 = line1[1] - a1 * line1[0]

    a2 = (line2[1] - line2[3]) / (line2[0] - line2[2])
    b2 = line2[1] - a2 * line2[0]

    if (abs(a1 - a2) < epsilon):
        return None

    x = (b2 - b1) / (a1 - a2)
    y = a1 * x + b1
    return (x, y)

### Processes Distance ###
# Returns the left and right distance arrays
def line_distances(lines):
    # compute left distances
    dist_dx = []
    for [x0, y0, x1, y1] in lines:
        dist_dx.append(distance(x0, y0, x1, y1))
    return np.array(dist_dx)

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        [x1, y1, x2, y2] = line        
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    return cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)