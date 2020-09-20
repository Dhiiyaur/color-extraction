import os
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE + '\data')



def import_image(BASE):

	path = BASE + '\data'
	dirs = os.listdir(path)

	# import image

	image_path = path + '\\' + dirs[0]

	# openCV image

	image = cv2.imread(image_path, cv2.IMREAD_COLOR)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	img = image.reshape((image.shape[0] * image.shape[1],image.shape[2]))    #reshape image into two dimensional

	return img

# helper function

def rgb_to_hex(color):
    return '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))


# helper function
def extraction(k, img):

	k = 5
	clt = KMeans(n_clusters = k)
	clt.fit(img)

	# count color cluster
	counts = Counter(clt.labels_)
	center_colors = clt.cluster_centers_   # color data, cluter center

	ordered_colors = [center_colors[i] for i in counts.keys()]

	# rgb to hex 

	hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]

	return hex_colors

def create_color_table(hex_colors):

	# create color table

	fig, ax = plt.subplots(figsize=(len(hex_colors) * 2, 2))
	x = 0
	for i in range(len(hex_colors)):
	    ax.add_patch(plt.Rectangle((x, 1), 1,1, color= hex_colors[i]))
	    ax.text(x + 0.05 , 0.85, hex_colors[i].upper())
	    x += 1

	plt.axis('off')
	ax.plot()   
	fig.savefig('result.png', dpi=100)


def main(BASE):

	img = import_image(BASE)
	hex_colors = extraction(5, img)
	create_color_table(hex_colors)

main(BASE)