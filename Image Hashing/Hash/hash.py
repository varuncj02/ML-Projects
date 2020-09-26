import numpy as np 
import cv2

def dhash(image , hashSize = 8):
    #Image made to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2GRAY)

    #Resizing the GrayScale into an n+1 * n format
    resized = cv2.resize(gray, (hashSize + 1, hashSize))

    #Computing the Gradient
    diff = resized[:, 1:] > resized[:, :-1]

    #Computing the Difference for Hash
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def convert_hash(h):
    #Converting Hash back to an integer from the VP-Tree
    return int(np.array(h, dtype="float64"))

#Hamming Distance between two different Hashes
def hamming_distance(x, y):
    return bin(int(x) ^ int(y)).count("1")
