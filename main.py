import numpy as np
import cv2
from PIL import Image
import math
from matplotlib import pyplot as plt

test_image = "test.png"

def open_image(image, resize = True):
    color_img = cv2.imread(image)
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    W, H = gray_img.shape
    if resize:
        gray_img = cv2.resize(gray_img, (int(W/2), int(H/2)))
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

    return c

def edge_detection(img_data):
    W, H = img_data.shape
    # for edge detection
    # low
    # KERNEL = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
    # # medium
    # KERNEL = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    # high
    KERNEL = np.flipud(np.fliplr(np.array([[-1,-1, -1], [-1, 8, -1], [-1, -1, -1]])))
    c = convo(img_data, KERNEL)
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

def sharp(img_data):
    W, H = img_data.shape
    # Sharpening image
    KERNEL =  np.flipud(np.fliplr(np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])))
    c = convo(img_data, KERNEL)
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

def gaussian_blur(img_data):
    W, H = img_data.shape
    # Gaussian Blue
    KERNEL =  np.flipud(np.fliplr(np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])))
    c = convo(img_data, KERNEL)
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


def sobel(img_data):
    W, H = img_data.shape
    KERNEL_vertical = np.flipud(np.fliplr(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])))
    KERNEL_horizontal = np.flipud(np.fliplr(np.array([[-1, -2, -1], [0, 0, 0], [1,2, 1]])))
    c = convo(img_data, KERNEL_vertical)
    final_convo = []
    final_convo_angles = []
    count = 0
    temp = []
    temp1 = []
    for k in c:
        count +=1
        Gx = np.sum(np.multiply(KERNEL_vertical, k))
        Gy = np.sum(np.multiply(KERNEL_horizontal, k))
        # print("Gx : ", Gx)
        # print("Gy : ", Gy)
        G = math.sqrt(Gx**2 + Gy**2)
        angle = np.rad2deg(np.arctan2(Gy, Gx))

        # print("G : ", G)
        if count <= W-2:
            temp.append(G)
            temp1.append(angle)
        else:
            final_convo.append(temp)
            final_convo_angles.append(temp1)
            count = 0
            temp = []
            temp1 = []

    final_convo = np.asarray(final_convo)
    final_convo_angles = np.asarray(final_convo_angles)
    # print("final_convo : ", final_convo_angles.shape )
    return final_convo, final_convo_angles

def non_maximum_suppression(image, angles):
    W, H = image.shape
    suppressed  = np.zeros((W, H))
    # print("suppressed : ", suppressed)
    for i in range(0, W - 1):
        for j in range(0, H - 1):
            if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                value_to_compare = max(image[i, j - 1], image[i, j + 1])
            elif (22.5 <= angles[i, j] < 67.5):
                value_to_compare = max(image[i - 1, j - 1], image[i + 1, j + 1])
            elif (67.5 <= angles[i, j] < 112.5):
                value_to_compare = max(image[i - 1, j], image[i + 1, j])
            else:
                value_to_compare = max(image[i + 1, j - 1], image[i - 1, j + 1])

            if image[i, j] >= value_to_compare:
                suppressed[i, j] = image[i, j]

    suppressed = np.multiply(suppressed, 255.0 / suppressed.max())
    return suppressed

def double_thresholding(image, high, low):
    weak = 50
    strong = 255
    size = image.shape
    result = np.zeros(size)
    weak_x, weak_y = np.where((image > low) & (image <= high))
    strong_x, strong_y = np.where(image >= high)
    result[strong_x, strong_y] = strong
    result[weak_x, weak_y] = weak
    dx = np.array((-1, -1, 0, 1, 1, 1, 0, -1))
    dy = np.array((0, 1, 1, 1, 0, -1, -1, -1))
    size = image.shape

    while len(strong_x):
        x = strong_x[0]
        y = strong_y[0]
        # print("x : "+ str(x) + " y : " + str(y))
        strong_x = np.delete(strong_x, 0)
        strong_y = np.delete(strong_y, 0)
        for direction in range(len(dx)):
            new_x = x + dx[direction]
            new_y = y + dy[direction]
            if((new_x >= 0 & new_x < size[0] & new_y >= 0 & new_y < size[1]) and (result[new_x, new_y]  == weak)):
                result[new_x, new_y] = strong
                np.append(strong_x, new_x)
                np.append(strong_y, new_y)
    result[result != strong] = 0
    return result


def main():
    image_data = open_image(test_image)
    # image_data = np.asarray([[105, 102, 100, 97, 96], [103, 99, 103, 101, 102], [101, 98, 104, 102, 100], [99, 101, 106, 104, 99], [104, 104, 104, 100, 98]])
    # print(image_data.shape)
    image_data = gaussian_blur(image_data)
    grad, angles = sobel(image_data)
    res = non_maximum_suppression(grad, angles)
    res = double_thresholding(res, 40, 0)
    img = Image.fromarray(res)
    img.show()


if __name__ == '__main__':
    main()
