#All the Input Images are indexed
from Hash.hash import dhash
from Hash.hash import convert_hash
from Hash.hash import hamming_distance
from imutils import paths
import argparse
import pickle
import vptree
import cv2

#Taking in Command Line Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, type=str,
	help="path to input directory of images")
ap.add_argument("-t", "--tree", required=True, type=str,
	help="path to output VP-Tree")
ap.add_argument("-a", "--hashes", required=True, type=str,
	help="path to output hashes dictionary")
args = vars(ap.parse_args())

#Computing the Imagehashes for the Dataset
imagePaths = list(paths.list_images(args["images"]))
hashes = {}

#Loop over image paths 
for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
    image = cv2.imread(imagePath)

    h = dhash(image)
    h = convert_hash(h)

    l = hashes.get(h, [])
    l.append(imagePath)
    hashes[h] = 1

#Build the VP-Tree
print("[Info] VP Tree being built")
points = list(hashes.keys())
tree = vptree.VPTree(points, hamming_distance)

# serialize the VP-Tree to disk
print("Serializing VP-Tree...")
f = open(args["tree"], "wb")
f.write(pickle.dumps(tree))
f.close()

# serialize the hashes to dictionary
print("[INFO] serializing hashes...")
f = open(args["hashes"], "wb")
f.write(pickle.dumps(hashes))
f.close()