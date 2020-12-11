import numpy as np
import cv2
from PIL import Image

test_image = "test.png"

def open_image(image):
    color_img = cv2.imread(image)
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("test ", gray_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return gray_img


def convo(aa, KERNEL):
    W, H = aa.shape
    k_size = len(KERNEL)
    off = k_size - 1

    padded_aa = np.pad(aa, 1, 'constant')

    w , h = padded_aa.shape
    c = []

    for i in range(0, w - off):
        for j in range(0, h - off):
            temp = padded_aa[ i : i + k_size  ,  j : j + k_size]
            if len(temp[0]) == k_size and len(temp) == k_size:
                c.append(padded_aa[ i : i + k_size  ,  j : j + k_size])

    cr = [np.sum(np.multiply(KERNEL, sub)) for sub in c]
    final_convo = []
    count = 0
    temp = []
    for l in range(len(cr)):
        count += 1
        if count <= W-1:
            temp.append(cr[l])
        else:
            final_convo.append(temp)
            count = 0
            temp = []


    final_convo = np.asarray(final_convo)
    # print("final_convo : ", final_convo.shape )
    return final_convo


def main():
    image_data = open_image(test_image)
    # image_data = np.asarray([[105, 102, 100, 97, 96], [103, 99, 103, 101, 102], [101, 98, 104, 102, 100], [99, 101, 106, 104, 99], [104, 104, 104, 100, 98]])
    # print(image_data.shape)

    # Sharpening image
    # KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    # for edge detection
    # low
    # KERNEL = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
    # # medium
    # KERNEL = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    # high
    # KERNEL = np.array([[-1,-1, -1], [-1, 8, -1], [-1, -1, -1]])

    # Gaussian Blue
    KERNEL = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])

    KERNEL = np.flipud(np.fliplr(KERNEL))

    sharp_img = convo(image_data, KERNEL)
    img = Image.fromarray(sharp_img)
    img.show()

if __name__ == '__main__':
    main()
