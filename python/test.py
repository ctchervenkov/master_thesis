import cv2
import numpy as np
from matplotlib import pyplot as plt


def gaussPyr(src,octaves):
	shape_origin = src.shape
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

nb_octaves = 8;

raw = cv2.imread('benchmark1.png') 

# Intensity image
r,g,b = cv2.split(raw)
I = (b+g+r)/3.0;

plt.figure()
plt.subplot(1,5,1),plt.imshow(raw), plt.title('raw')
plt.subplot(1,5,2),plt.imshow(r, cmap='gray'), plt.title('r')
plt.subplot(1,5,3),plt.imshow(g, cmap='gray'), plt.title('g')
plt.subplot(1,5,4),plt.imshow(b, cmap='gray'), plt.title('b')
plt.subplot(1,5,5),plt.imshow(I, cmap='gray'), plt.title('I')
# plt.show()


# Normalize r,g,b by I & create color channels
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


# print len(I_gp), len(R_gp), len(G_gp), len(B_gp), len(Y_gp)


# printPyr(I_gp)
# plt.show();


# Build oriented Gabor pyramids at 0, 45, 90 & 135 degrees
plt.figure();

thetas = np.array([0,45,90,135])/180.0*np.pi;
O_gp = np.ndarray(shape=(thetas.size,nb_octaves),dtype=object);

for t in range(thetas.size):
	theta = thetas[t];
	gabor_kernel = cv2.getGaborKernel((5,5), 1, theta, 1, 1, 0);
	gabor_kernel = gabor_kernel/np.sum(gabor_kernel);
	# print gabor_kernel

	src = I;
	# print src.shape
	O_gp[t,0] = src;
	plt.subplot(4,8,8*t+1),plt.imshow(O_gp[t,0], cmap='gray')
	
	for i in range(1,nb_octaves):
		# print theta, i
		
		src = cv2.filter2D(src, -1, gabor_kernel);

		#Downsample step
		src = cv2.resize(src, (src.shape[1]/2,src.shape[0]/2),interpolation=cv2.INTER_AREA)
		# src_temp = np.zeros((src.shape[0]/2,src.shape[1]/2));

		# print src.shape[0], src.shape[1]
		# print src.shape[0]/2, src.shape[1]/2

		# for y in range(src.shape[0]/2):
		# 	for x in range(src.shape[1]/2):
		# 		src_temp = (src[2*y,2*x] + src[2*y+1,2*x] + src[2*y,2*x+1] + src[2*y+1,2*x+1])/4.0

		# print src_temp.shape_origin

		# src = src_temp;
		O_gp[t,i] = src;
		plt.subplot(4,8,8*t+i+1),plt.imshow(O_gp[t,i], cmap='gray')

plt.show();




# plt.figure()

# shape_origin = I.shape
# print shape_origin

# # create intensity Gaussian pyramid
# igp = np.zeros((shape_origin[0],shape_origin[1],nb_octaves));
# cgp = np.zeros((shape_origin[0],shape_origin[1],nb_octaves));
# ogp = np.zeros((shape_origin[0],shape_origin[1],nb_octaves));

# for i in range (0,nb_octaves):

# 	I = cv2.pyrDown(I)
# 	igp[:,:,i] = cv2.resize(I,(shape_origin[1],shape_origin[0]),interpolation = cv2.INTER_LINEAR);
# 	plt.subplot(4,4,i+1), plt.imshow(igp[:,:,i],cmap='gray'), plt.title('RAW')

# 	print I.shape
# 	print igp[:,:,i].shape

# plt.show()

# # create intensity Gaussian pyramid