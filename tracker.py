### Line Finding Pipeline ###
class Tracker():
    
    # Initialises
    # - window_width: sliding window width
    # - window_height: sliding window height
    # - window_margin: sliding window margin
    # - smooth_factor: timesteps for averaging lane lines
    # - theta: polynomial term for regression functions
    # - threshold: value for thresholding the maximal in coordinate descent
    # - gain: gain value to keep lanes adjacent when tracking
    # - alpha: first smoothing value for minimising path errors
    # - beta: second smoothing value for minimising path errors
    def __init__(self, window_width, window_height, window_margin, smooth_factor = 3, \
                 theta = 2, threshold = 0.0001, gain = 0.9, alpha = 0.5, beta = 0.5, \
                 my_per_pix = 30 / 720, mx_per_pix = 3.7 / 700):
        self.window_width = window_width
        self.window_height = window_height
        self.window_margin = window_margin
        self.smooth_factor = smooth_factor
        self.theta = theta
        self.threshold = threshold
        self.gain = gain
        self.alpha = alpha
        self.beta = beta
        
        self.MY_per_pix = my_per_pix
        self.MX_per_pix = mx_per_pix
        
        self.left_lane = None
        self.right_lane = None
        
        # used for keeping track of smoothed lines
        self.counter = 0
    
    # Finds and detects the adjacent left and right lane lines in the given image scene.
    # - img: Raw input image in RGB colourspace.
    # Returns: 2D Tuple of left and right lane line objects (see Line class).
    def detect_lanes(self, img):
        # mask the input image
        mask = lane_line_mask(img, sobel_kernels, sobel_abs_thresh, sobel_mag_thresh, sobel_dir_thresh)
        # perspective transform
        pers_mask = camera.warp(mask)
        # compute the window centroids
        centroids = find_window_centroids(pers_mask, self.window_width, self.window_height, self.window_margin, \
                                          threshold = self.threshold, gain = self.gain, theta = self.theta)
        # smoothed centroids
        centroids_smooth = smooth_path(centroids[:,[0,4]], beta = self.beta, alpha = self.alpha)
        # construct y range
        y_vals = np.arange(0, mask.shape[0])
        y_vals_res = np.arange(mask.shape[0]-(self.window_height/2), 0, -self.window_height)
        # compute left line coefficients
        left_coef = np.polyfit(y_vals_res, centroids_smooth[:,0], self.theta)
        left_fit = left_coef[0] * y_vals * y_vals + left_coef[1] * y_vals + left_coef[2]
        left_fit = np.array(left_fit, np.int32)
        
        # compute right line coefficients
        right_coef = np.polyfit(y_vals_res, centroids_smooth[:,1], self.theta)
        right_fit = right_coef[0] * y_vals * y_vals + right_coef[1] * y_vals + right_coef[2]
        right_fit = np.array(right_fit, np.int32)
        # window offset
        offset_width = window_width / 2
        
        # construct left and right Line objects
        left = Line(left_coef, left_fit, y_vals, offset_width)
        right = Line(right_coef, right_fit, y_vals, offset_width)
        
        return (left, right)
    
    # Tracks and smoothes the lane lines detected from the previous detection
    # - left: Line object
    # - right: Line object
    def track(self, left, right):
        if (self.counter == 0):
            self.left_lane = left
            self.right_lane = right
        else:
            # average the lanes for n
            self.left_lane.smooth(left, self.smooth_factor)
            self.right_lane.smooth(right, self.smooth_factor)
        
        # increment counter for averaging
        self.counter += 1

    # Take input lane line and return the curvature of the line.
    # - lane: Lane line to compute corresponding curvature
    def curvature(self, left, right):
        # compute curvature of line (self)
        curve_x = np.average([left.mean_xvals, right.mean_xvals], axis = 0)
        curve_y = np.average([left.Y_vals, right.Y_vals], axis = 0)
        curve_fit = np.polyfit(curve_y, curve_x, self.theta)
    
        x_val = curve_x[::-1][0]
        line_rad = ((1. + (2. * curve_fit[0] * x_val * self.MY_per_pix + curve_fit[1]) ** 2) ** 1.5)\
                    / np.absolute(2 * curve_fit[0])
        return line_rad

    # Compute the distance from center using the current detected lines
    # - left: Left lane (Line object)
    # - right: Right lane (Line object)
    # - width: Image width
    def center_distance(self, left, right, width):
        center_offset = (left.X_vals[-1] + right.X_vals[-1]) / 2.
        camera_dist = (center_offset - width / 2.) * self.MX_per_pix
        return camera_dist
    
    # Resets the tracking state and clears any lane memory
    def reset(self):
        self.counter = 0
    
    def draw_lanes(self, img, left, right):
        # initialise lane drawing layers
        road_img = np.zeros_like(img)
        road_imgbg = np.zeros_like(img)
        
        l_fit = left.fit()
        r_fit = right.fit()
        
        # fill polygons
        cv2.fillPoly(road_img, [l_fit], color = [0,0,255])
        cv2.fillPoly(road_img, [r_fit], color = [255,0,0])
        cv2.fillPoly(road_imgbg, [l_fit], color = [255,255,255])
        cv2.fillPoly(road_imgbg, [r_fit], color = [255,255,255])
        
        return (road_img, road_imgbg)
        
# Define a class to receive the characteristics of each line detection
class Line():
    # Initialises
    #  - coefficients: 1D array of polynomial coefficients
    #  - xvals: X points of the fitted polynomial
    #  - yvals: Y points of the fitted polynomial
    def __init__(self, coefficients, xvals, yvals, offset):
        # was the line detected in the last iteration?
        self.detected = True
        # x values of the last n fits of the line
        self.recent_xvals = [xvals]
        # average x values of the fitted line over the last n iterations
        self.mean_xvals = xvals
        # recent polynomial coefficients
        self.recent_coef = [coefficients]
        # polynomial coefficients averaged over the last n iterations
        self.mean_coef = coefficients
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        # x values for detected line pixels
        self.X_vals = xvals
        # y values for detected line pixels
        self.Y_vals = yvals
        # offset
        self.offset = offset
    
    # Returns the fitted x,y points for drawing
    def fit(self):
        return np.array(list(zip(np.concatenate((self.mean_xvals - self.offset, self.mean_xvals[::-1] + self.offset), axis = 0),\
                                  np.concatenate((self.Y_vals, self.Y_vals[::-1]), axis = 0))), np.int32)
    # Smoothes the current line with the supplied line.
    #  - line: Line object to append and smooth
    #  - max_count: Maximum memory count used in smoothing
    def smooth(self, line, max_count):
        self.detected = True
        
        # smooth xvals over {0 < len <= n}
        self.recent_xvals.append(line.X_vals)
        self.recent_xvals = self.recent_xvals[-max_count:]
        self.mean_xvals = np.sum(self.recent_xvals, axis = 0) / len(self.recent_xvals)
        
        # smooth coefficients over {0 < len <= n}
        self.recent_coef.append(line.recent_coef[-1])
        self.recent_coef = self.recent_coef[-max_count:]
        self.mean_coef = np.sum(self.recent_coef, axis = 0) / len(self.recent_coef)
        
        # store additional vars
        self.diffs = self.recent_coef[-1] - line.recent_coef[-1]
        self.X_vals = line.X_vals
        self.Y_vals = line.Y_vals