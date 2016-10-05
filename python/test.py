import cv2
import numpy as np
from matplotlib import pyplot as plt

## Function definitions ##
def getIntensityImage(raw):
	b,g,r = cv2.split(raw)
	I = (b+g+r)/3.0;
	return I, r, g, b;

def getColorChannels(r,b,g,I):
	I_max = np.amax(I);
	for i in range(I.shape[0]):
		for j in range(I.shape[1]):
			if I[i,j] > I_max/10:
				r[i,j] = r[i,j]/I[i,j]
				g[i,j] = g[i,j]/I[i,j]
				b[i,j] = b[i,j]/I[i,j]
			else:
				r[i,j] = 0
				g[i,j] = 0
				b[i,j] = 0

	R = r - (g+b)/2.0;
	G = g - (r+b)/2.0;
	B = b - (r+g)/2.0;
	Y = (r+g)/2.0 - np.absolute(r-g)/2.0 - b;

	R = np.maximum(0,R);
	G = np.maximum(0,G);
	B = np.maximum(0,B);
	Y = np.maximum(0,Y);

	return R, G, B, Y;

def buildPyramid(src,kernel,octaves):
	pyr = [src];
	for i in range(octaves):
		# Filter, downsample and append result
		src = cv2.filter2D(src, -1, kernel);
		src = cv2.resize(src, (src.shape[1]/2,src.shape[0]/2),interpolation=cv2.INTER_AREA)
		pyr.append(src);
	return gp;

def gaussPyr(src,octaves):
	gp = [src];
	for i in range(octaves):
		src = cv2.pyrDown(src)
		gp.append(src);
	return gp;

def printPyr(gp):
	plt.figure()
	for i in range(len(gp)):
		plt.subplot(1,len(gp),i+1)
		plt.imshow(gp[i], cmap='gray')

def centerSurroundDiff(im_c,im_s):
	im_cs = np.zeros(im_c.shape);
	im_s = cv2.resize(im_s, (im_c.shape[1],im_c.shape[0]),interpolation=cv2.INTER_LINEAR);
	im_cs = np.absolute(im_c - im_s);
	return im_cs;

def mapNormalization(src):
	if np.absolute(np.amax(src)) > 0.0:
		src = src / np.amax(src) * 100.0;
	minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(src);
	M = maxVal;
	threshold = 0.5*(maxVal - minVal) + minVal;
	m = 0;
	nb_peaks = 0;

	while maxVal > threshold:
		maxVal = np.amax(src[src<maxVal]);
		m += maxVal;
		nb_peaks += 1;

	if nb_peaks > 0:
		m /= nb_peaks;
	# print M, m
	out = src * np.power(M-m,2);

	return out;

def buildIntensityConspicuityMap(I_pyr,c_list,delta_list):
	reduce_scale = np.amax(c_list);
	consp_map = np.zeros(I_pyr[reduce_scale].shape);
	for c in c_list:
		for delta in delta_list:
			s = c + delta;
			consp_map += cv2.resize(mapNormalization(centerSurroundDiff(I_pyr[c],I_pyr[s])), (consp_map.shape[1],consp_map.shape[0]),interpolation=cv2.INTER_LINEAR);

	return consp_map;

def buildColorConspicuityMap(R_pyr,G_pyr,B_pyr,Y_pyr,c_list,delta_list):
	reduce_scale = np.amax(c_list);
	consp_map = np.zeros(R_pyr[reduce_scale].shape);
	for c in c_list:
		for delta in delta_list:
			s = c + delta;
			RG_cs = centerSurroundDiff(R_pyr[c] - G_pyr[c], G_pyr[s] - R_pyr[s]);
			BY_cs = centerSurroundDiff(B_pyr[c] - Y_pyr[c], Y_pyr[s] - B_pyr[s]);
			norm_im = mapNormalization(RG_cs) + mapNormalization(BY_cs); 
			consp_map += cv2.resize(norm_im, (consp_map.shape[1],consp_map.shape[0]),interpolation=cv2.INTER_LINEAR);

	return consp_map;

# TODO :

def gaborPyr(src,nb_octaves,thetas):
	# plt.figure();
	gp = np.ndarray(shape=(thetas.size,nb_octaves),dtype=object);

	for t in range(thetas.size):
		theta = thetas[t];
		gabor_kernel = cv2.getGaborKernel((5,5), 1, theta, 1, 1, 0);
		gabor_kernel = gabor_kernel/np.sum(gabor_kernel);


		gp[t,0] = src;
		# plt.subplot(4,8,8*t+1),plt.imshow(gp[t,0], cmap='gray')
		for i in range(1,nb_octaves):

			src = cv2.filter2D(src, -1, gabor_kernel);

			#Downsample step
			src = cv2.resize(src, (src.shape[1]/2,src.shape[0]/2),interpolation=cv2.INTER_AREA)

			gp[t,i] = src;
			# plt.subplot(4,8,8*t+i+1),plt.imshow(gp[t,i], cmap='gray')

	# plt.show();
	return gp

# def buildOrientationConspicuityMap:




## MAIN TEST CODE ##

nb_octaves = 8;

raw = cv2.imread('benchmark3.png') 

# Intensity image
I, r, g, b = getIntensityImage(raw);

plt.figure()
plt.subplot(1,5,1),plt.imshow(raw), plt.title('raw')
plt.subplot(1,5,2),plt.imshow(r, cmap='gray'), plt.title('r')
plt.subplot(1,5,3),plt.imshow(g, cmap='gray'), plt.title('g')
plt.subplot(1,5,4),plt.imshow(b, cmap='gray'), plt.title('b')
plt.subplot(1,5,5),plt.imshow(I, cmap='gray'), plt.title('I')
# plt.show()

# Normalize r,g,b by I & create color channels
R, G, B, Y = getColorChannels(r,b,g,I);

plt.figure()
plt.subplot(1,4,1),plt.imshow(R, cmap='gray'), plt.title('R')
plt.subplot(1,4,2),plt.imshow(G, cmap='gray'), plt.title('G')
plt.subplot(1,4,3),plt.imshow(B, cmap='gray'), plt.title('B')
plt.subplot(1,4,4),plt.imshow(Y, cmap='gray'), plt.title('Y')
# plt.show()

# Build Gaussian pyramids
I_gp = gaussPyr(I,nb_octaves);
R_gp = gaussPyr(R,nb_octaves);
G_gp = gaussPyr(G,nb_octaves);
B_gp = gaussPyr(B,nb_octaves);
Y_gp = gaussPyr(Y,nb_octaves);

# TODO: Build Gabor pyramid

# # Build oriented Gabor pyramids at 0, 45, 90 & 135 degrees
thetas = np.array([0.0,45.0,90.0,135.0]) * np.pi / 180.0;
O_gp = gaborPyr(I,nb_octaves,thetas);


# Build conspicuity maps
I_bar = buildIntensityConspicuityMap(I_gp,np.array([2,3,4]),np.array([3,4]))
C_bar = buildColorConspicuityMap(R_gp,G_gp,B_gp,Y_gp,np.array([2,3,4]),np.array([3,4]))

plt.figure()
plt.subplot(1,3,1),plt.imshow(I_bar, cmap='gray'), plt.title('Conspicuity Map Intensity')
plt.subplot(1,3,2),plt.imshow(C_bar, cmap='gray'), plt.title('Conspicuity Map Color')

# plt.show();
# print I_bar.shape, C_bar.shape

#TODO: Build orientation conspicuity map
O_bar = np.zeros(I_bar.shape);
S = (I_bar + C_bar + O_bar)/3.0;


plt.figure()
plt.imshow(S, cmap='gray'), plt.title('Saliency')
plt.show();