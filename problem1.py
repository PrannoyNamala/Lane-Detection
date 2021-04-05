import cv2
import numpy as np

total_no_of_pixels = 1920 * 1080


# User defined Histogram equalization function
def channel_equalzation(v):
    histogram = dict(zip(range(0, 256), np.zeros(256, np.uint)))  # Histogram is initialized

    for i in v:
        for j in i:
            histogram[j] += 1

    cdf = {}
    sum_of_hist = 0
    for j in range(0, 256):
        sum_of_hist += histogram[j]
        cdf[j] = np.uint8(255 * (sum_of_hist / (v.shape[1] * v.shape[0])))

    for i in range(0, v.shape[1]):
        for j in range(0, v.shape[0]):
            v[j][i] = cdf[v[j][i]]

    return v


# User defined function for Gamma Correction
def gamma_correction(v, gamma):
    o = np.divide(v, 255)

    o = np.power(o, 1 / gamma)

    scale_ = 255 / np.amax(o)

    v_new = np.multiply(o, scale_)

    return v_new.astype(np.uint8)


# Implementation from Opencv tutorials
def new_func(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    cdf = hist.cumsum()
    # cdf_normalized = cdf * hist.max() / cdf.max()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    return cdf[img]


# CLAHE Implementation from Opencv tutorials
def chale(v):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(v)
    return cl1


# This function is to compare between different approaches for equalization
def hist_equalization_color(video_name='Night Drive - 2689.mp4', fps=25, name_of_output=None):
    if type(video_name):
        cap = cv2.VideoCapture(video_name)  # Taking input of the video
    else:
        print("Check the video name")
        return False, None

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(name_of_output + '.avi', fourcc, fps, (1920, 1080), 0)

    while True:
        a, frame = cap.read()

        if not a:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # converting color spaces

        h, s, v = cv2.split(hsv)

        v_ = cv2.equalizeHist(v)  # Inbuilt Function implementation

        v_ud = gamma_correction(channel_equalzation(v), 2)  # User defined function implementation with gamma correction

        v_ot = new_func(v)  # Opencv Tutorials implementation

        v_chale = gamma_correction(chale(v), 5)  # CHAlE with gamma correction implementation

        img_if = cv2.cvtColor(cv2.merge((h, s, v_)), cv2.COLOR_HSV2BGR)  # Merging all images

        img_ud = cv2.cvtColor(cv2.merge((h, s, v_ud)), cv2.COLOR_HSV2BGR)

        img_ot = cv2.cvtColor(cv2.merge((h, s, v_ot)), cv2.COLOR_HSV2BGR)

        img_chale = cv2.cvtColor(cv2.merge((h, s, v_chale)), cv2.COLOR_HSV2BGR)

        b, g, r = cv2.split(frame)

        b = channel_equalzation(b)

        g = channel_equalzation(g)

        r = channel_equalzation(r)

        op_img = cv2.merge((b, g, r))  # Image with each color channel equalization

        divider = np.zeros((1080, 10, 3), np.uint8)

        color_res = np.hstack((frame, divider, img_if, divider, img_ud, divider, img_ot, divider, img_chale))

        cv2.imshow("Compare Image", color_res)
        cv2.waitKey(1)


# Main function to run
def hist_equalization(video_name='Night Drive - 2689.mp4', fps=25, name_of_output=None):
    if type(video_name):
        cap = cv2.VideoCapture(video_name)  # Taking input of the video
    else:
        print("Check the video name")
        return False, None
    # Test image pipeline
    image = cv2.imread("rgb-bw-b.png")

    im_g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    im_g_hist = channel_equalzation(im_g)

    cv2.imshow("CO", im_g_hist)
    cv2.waitKey(0)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(name_of_output + '.avi', fourcc, fps, (1920, 1080), 0)

    while True:
        a, frame = cap.read()

        if not a:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray_ud = channel_equalzation(gray)

        out.write(gray_ud)
