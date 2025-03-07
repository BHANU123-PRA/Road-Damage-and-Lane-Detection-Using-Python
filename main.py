from flask import *  
from flask import Flask, render_template, Response
import pandas as pd
import numpy as np
import os
from camera import VideoCamera
import pickle
import cv2
import glob
from camera import VideoCamera

from random import randint

import matplotlib.pylab as plt
import cv2
import numpy as np
import time
from flask import Flask, render_template, Response



def fit_poly(img_shape, leftx, lefty, rightx, righty):
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	# Generate x and y values for plotting
	ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	return left_fitx, right_fitx, ploty


IMAGE_FOLDER = 'static/'
PROCESSED_FOLDER = 'static/processed/'
#IMAGE_FOLDER = os.path.join('upload', 'images')


app = Flask(__name__)  
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER



@app.route('/', methods=["GET", "POST"])
@app.route('/first')
def first():
    return render_template('first.html')
@app.route('/login')
def login():
    return render_template('login.html')



@app.route('/upload')  
def upload():
	return render_template("file_upload_form.html")  
 
@app.route('/success', methods = ['POST'])
def success():
	if request.method == 'POST':
		f = request.files['file']
		#hls = (np.float32(image), cv2.COLOR_RGB2HLS)
		f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
		#print(f.filename)
		full_filename = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
		#filepath = os.path.join(app.config['imgdir'], filename);
		#file.save(filepath)
		image_ext = cv2.imread(full_filename)
		initial_image = np.copy(image_ext)

		objp = np.zeros((6*9,3), np.float32)
		objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

		# Arrays to store object points and image points from all the images.
		objpoints = [] # 3d points in real world space
		imgpoints = [] # 2d points in image plane.

		# Make a list of calibration images
		images_for_calibration = glob.glob('camera_cal/calibration*.jpg')

		# Step through the list and search for chessboard corners
		for f_name in images_for_calibration:
	    		img_read = cv2.imread(f_name)
    			gray = cv2.cvtColor(img_read,cv2.COLOR_BGR2GRAY)

    			# Find the chessboard corners
    			ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    			# If found, add object points, image points
    			if ret == True:
        			objpoints.append(objp)
        			imgpoints.append(corners)


		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, initial_image.shape[1::-1], None, None)
		undistorted = cv2.undistort(initial_image, mtx, dist, None, mtx)
		hls = cv2.cvtColor(undistorted, cv2.COLOR_RGB2HLS)
		s_channel = hls[:,:,2]
		gray = cv2.cvtColor(undistorted, cv2.COLOR_RGB2GRAY)
		
		sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
		abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
		scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
		thresh_min = 20
		thresh_max = 100
		sxbinary = np.zeros_like(scaled_sobel)
		sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
		s_thresh_min = 170
		s_thresh_max = 255
		s_binary = np.zeros_like(s_channel)
		s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
		color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
		combined_binary = np.zeros_like(sxbinary)
		combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

		nx = 9 # the number of inside corners in x
		ny = 6 # the number of inside corners in y
		src = np.float32([[590,450],[687,450],[1100,720],[200,720]])
		dst = np.float32([[300,0],[900,0],[900,720],[300,720]])
		
		im_size = (combined_binary.shape[1], combined_binary.shape[0])
		M = cv2.getPerspectiveTransform(src, dst)
		M_inverse = cv2.getPerspectiveTransform(dst, src)

		warped_image = cv2.warpPerspective(combined_binary, M, im_size, flags=cv2.INTER_NEAREST)
		left_fit = np.array([ 2.13935315e-04, -3.77507980e-01,  4.76902175e+02])
		right_fit = np.array([4.17622148e-04, -4.93848953e-01,  1.11806170e+03])
		
		margin = 100
		nonzero = warped_image.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])



		left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
		right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds]
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds]
		left_fitx, right_fitx, ploty = fit_poly(warped_image.shape, leftx, lefty, rightx, righty)
		warp_zero = np.zeros_like(warped_image).astype(np.uint8)
		color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
		pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
		pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
		pts = np.hstack((pts_left, pts_right))
		cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
		newwarp = cv2.warpPerspective(color_warp, M_inverse, im_size)
		
		result_final = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
		
		output_image_after_detecting = result_final
		
		i = randint(1, 1000000)
		char = str(i)
		hls_name = 'sample_'+char+'.jpg'
		cv2.imwrite('static/processed/'+hls_name, output_image_after_detecting)
		full_filename_processed = os.path.join(app.config['PROCESSED_FOLDER'], hls_name)		
		final_text = 'Results after Detecting Lane Area over Input Image'
		return render_template("success.html", name = final_text, img_in = full_filename, img = full_filename_processed)
		

@app.route('/info', methods = ['POST'])
def info():
	if request.method == 'POST':
		return render_template("info.html")


@app.route('/index')
def index():
    # Video streaming home page.
    return render_template('index.html')


def roi(img, vertices):
    mask = np.zeros_like(img)
    # channel_count = img.shape[2]
    match_mask_color = (255)
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# drawing lines as lanes detected

def draw_lines(img, lines):
    img = np.copy(img)
    line_image = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8) 
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), thickness=4)
            # cv2.fillPoly(line_image, pts = ((x1,y1), (x2,y2)), color = (0, 255,120))
    img = cv2.addWeighted(img, 0.8, line_image, 1, 0.0)        
    return img

# image = cv2.imread('road5.png')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# process includes
# 1. Grascaling the image
# 2. Canny Edge Detection
# 3. Cropping Image acc to roi 
# 4. Drawing lines on the cropped image
# 5. Super imposing it on original image using BITWISE AND

def process(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    # (704, 1279, 3)

    region_of_interest_vertices = [
        (0, height),
        (width/2, height/2),
        (width, height)
    ]


    gray_cropped_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cannyImage = cv2.Canny(gray_cropped_image, 100, 120)
    cropped_image = roi(cannyImage, np.array([region_of_interest_vertices], np.int32))

    lines = cv2.HoughLinesP(cropped_image, 
                            rho = 2, 
                            theta = np.pi/60, 
                            threshold = 160, 
                            lines=np.array([]), 
                            minLineLength = 40, 
                            maxLineGap=25)

    imageWithLines = draw_lines(image, lines)
    # colorAddition = cv2.fillPoly(imageWithLines, np.array([region_of_interest_vertices], np.int32), (120,200,200))

    return imageWithLines

# plt.imshow(imageWithLines)
# plt.show()

# rescaling frame due to different sizes

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

# cap.release()
# cv2.destroyAllWindows()



def gen():
    """Video streaming generator function."""
    cap = cv2.VideoCapture('project_video.mp4')

    # Read until video is completed
    while(cap.isOpened()):
        ret, frame = cap.read()  # import image
        if not ret: #if vid finish repeat
            frame = cv2.VideoCapture("project_video.mp4")
            continue
        if ret:  # if there is a frame continue with code
            frame = process(frame)
            frame = rescale_frame(frame, percent=100)
            image = cv2.resize(frame, (0, 0), None, 1, 1)  # resize image
            
        frame = cv2.imencode('.jpg', image)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        #time.sleep(0.1)
        key = cv2.waitKey(20)
        if key == 27:
           break
   
        

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/index1')
def index1():
    return render_template('index1.html')



@app.route('/chart')
def chart():
    return render_template('chart.html')


if __name__ == '__main__':  
	app.run(host="127.0.0.1",port=8080,debug=True)  






