from PIL import Image
import numpy as np
import cv2

test_image = "test.png"

def open_image(image):
    color_img = cv2.imread(image)
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('TopLeft5x5.jpg', gray_img[:5, :5])
    # cv2.imshow("test ", gray_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return gray_img


def convo(img, k):
    print("image : ", img)
    print("kerne l ;", k)


def main():
    image_data = open_image(test_image)
    KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    KERNEL = np.flipud(np.fliplr(KERNEL))

    convo(image_data[:5, :5], KERNEL)

if __name__ == '__main__':
    main()
