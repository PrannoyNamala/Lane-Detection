import cv2
import os
import yaml
import numpy as np

# Function to load images from a folder in a order
def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

# Selects a triangular region for video one
def triangle_roi1(image):
    new_image = np.zeros_like(image)
    vertices = np.array(
        ((0, image.shape[0] - 1), ((image.shape[1] / 2) - 100, (image.shape[0] / 2) + 10),
         (image.shape[1] - 1, image.shape[0] - 1)),
        dtype=np.int32)
    cv2.fillPoly(new_image, [vertices], (255, 255, 255))
    return cv2.bitwise_and(image, new_image)

# Selects a triangular region for video two
def triangle_roi2(image):
    new_image = np.zeros_like(image)
    vertices = np.array(
        ((0, image.shape[0] - 1), ((image.shape[1] / 2) + 50, (image.shape[0] / 2) + 100),
         (image.shape[1] - 1, image.shape[0] - 1)),
        dtype=np.int32)
    cv2.fillPoly(new_image, [vertices], (255, 255, 255))
    return cv2.bitwise_and(image, new_image)

# Loads parameters from the YAML file
def parameter_selector(parameters_):
    parameters = yaml.load(parameters_, Loader=yaml.FullLoader)

    k = parameters['K'].split()

    mtx = np.array(k, np.float128).reshape(3, 3).astype(np.int32)

    dist = parameters['D'].split()

    dist = np.array(dist, np.float128).astype(np.int32)

    return mtx, dist

# Lane estimate function for data 1
def lane_estimates(mask, direction='straight'):
    high_values = []

    for col in range(0, mask.shape[1]):
        s = mask[:, col]
        count = 0
        for val in s:
            if val != 0:
                count += 1
        high_values.append(count)

    left_high = high_values.index(max(high_values[0:int(len(high_values) / 2)]))
    right_high = high_values.index(max(high_values[int(len(high_values) / 2):len(high_values)]))

    point_dic = {'left': np.transpose(np.nonzero(mask[:, left_high - 2:left_high + 3])),
                 'right': np.transpose(np.nonzero(mask[:, right_high - 2:right_high + 3]))}
    point_dic["left"][:, 1] += left_high - 2
    point_dic["right"][:, 1] += right_high - 2
    left_x = point_dic["left"][:, 0]
    left_y = point_dic["left"][:, 1]
    right_x = point_dic["right"][:, 0]
    right_y = point_dic["right"][:, 1]

    allpoints = np.array(
        ((left_y[0], right_x[0]), (left_y[-1], right_x[-1]), (right_y[-1], right_x[-1]), (right_y[0], right_x[0])),
        dtype=np.int32)

    lane_equation_fit = {"left": np.polyfit(left_y, left_x, 2), "right": np.polyfit(right_y, right_x, 2)}

    right_slope = - right_x[-1] - lane_equation_fit["right"][1] / (2 * lane_equation_fit["right"][0]) / (
                (-(lane_equation_fit["right"][1]) ** 2 / (2 * lane_equation_fit["right"][0])) +
                lane_equation_fit["right"][2] - right_y[-1])  # x and y as in opencv

    if np.abs(right_slope) - 200 < 11:
        direction = 'right'
    elif np.abs(right_slope) - 200 > 17:
        direction = 'straight'
    elif np.abs(right_slope) - 200 < 17:
        direction = 'left'
    else:
        pass
    return lane_equation_fit, allpoints, direction

# Implementation for data_1
def impl():
    images = load_images_from_folder("/home/prannoy/s21subbmissions/673/Project 2/data_1/data")

    parameters_ = open("/home/prannoy/s21subbmissions/673/Project 2/data_1/camera_params.yaml")

    mtx, dist = parameter_selector(parameters_)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('video1p.avi', fourcc, 30, (images[1].shape[1], images[1].shape[0]))


    for img in images:
        dst = cv2.undistort(img, mtx, dist, None, mtx)

        dst = cv2.fastNlMeansDenoisingColored(dst, None, 3, 10, 7, 21)

        dst_orig_crp = dst[240:-1, :]

        dst_triangle = triangle_roi1(dst)

        dst_crop = dst_triangle[240:-1, :]

        pts_1 = np.array(((150, 250), (850, 230), (730, 80), (505, 55)), dtype=np.float32)
        pts_2 = np.array(((0, 200), (50, 200), (50, 0), (0, 0)), dtype=np.float32)

        h = cv2.getPerspectiveTransform(pts_1, pts_2)

        new_img = cv2.warpPerspective(dst_crop, h, (70, 250))

        new_img_hsv = cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(new_img_hsv, (0, 0, 184), (255, 255, 255))

        birdeye_equation, birdeye_lanepoints, direction = lane_estimates(mask, h)

        lane_mask = np.zeros_like(new_img)

        cv2.fillPoly(lane_mask, [birdeye_lanepoints], (255, 0, 0))

        cv2.line(lane_mask, (birdeye_lanepoints[0, 0], birdeye_lanepoints[0, 1]), (birdeye_lanepoints[1, 0], birdeye_lanepoints[1, 1]), (0,0,255), 10)

        cv2.line(lane_mask, (birdeye_lanepoints[-2, 0], birdeye_lanepoints[-2, 1]), (birdeye_lanepoints[-1, 0], birdeye_lanepoints[-1, 1]), (0, 0, 255), 10)

        lane_img = cv2.warpPerspective(lane_mask, np.linalg.inv(h), (dst_orig_crp.shape[1], dst_orig_crp.shape[0]))

        dst_final_crop = cv2.addWeighted(dst_orig_crp, 0.7, lane_img, 0.3, 0)

        dst[240:-1, :] = dst_final_crop

        dst = cv2.putText(dst, direction, (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        out.write(dst)

# Estimation function for data_2
def lane_estimates2(mask, direction='straight'):
    high_values = []

    for col in range(0, mask.shape[1]):
        s = mask[:, col]
        count = 0
        for val in s:
            if val != 0:
                count += 1
        high_values.append(count)

    left_high = high_values.index(max(high_values[0:int(len(high_values) / 2)]))
    right_high = high_values.index(max(high_values[int(len(high_values) / 2):len(high_values)]))

    point_dic = {'left': np.transpose(np.nonzero(mask[:, left_high - 2:left_high + 3])),
                 'right': np.transpose(np.nonzero(mask[:, right_high - 2:right_high + 3]))}
    point_dic["left"][:, 1] += left_high - 2
    point_dic["right"][:, 1] += right_high - 2
    left_x = point_dic["left"][:, 0]
    left_y = point_dic["left"][:, 1]
    right_x = point_dic["right"][:, 0]
    right_y = point_dic["right"][:, 1]

    allpoints = np.array(
        ((left_y[0], left_x[0]), (left_y[-1], left_x[-1]), (right_y[-1], left_x[-1]), (right_y[0], left_x[0])),
        dtype=np.int32)

    lane_equation_fit = {"left": np.polyfit(left_y, left_x, 2), "right": np.polyfit(right_y, right_x, 2)}

    left_slope = - left_x[-1] - lane_equation_fit["left"][1] / (2 * lane_equation_fit["left"][0]) / (
                (-(lane_equation_fit["left"][1]) ** 2 / (2 * lane_equation_fit["left"][0])) +
                lane_equation_fit["left"][2] - left_y[-1])  # x and y as in opencv

    if np.abs(left_slope) - 200 < 11:
        direction = 'right'
    elif np.abs(left_slope) - 200 > 17:
        direction = 'straight'
    elif np.abs(left_slope) - 200 < 17:
        direction = 'left'
    else:
        pass
    return lane_equation_fit, allpoints, direction

# Function for implementation on data_2
def impl2():
    cap = cv2.VideoCapture("data_2/challenge_video.mp4")
    parameters_ = open("/home/prannoy/s21subbmissions/673/Project 2/data_2/cam_params.yaml")

    mtx, dist = parameter_selector(parameters_)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('video2p.avi', fourcc, 30, (1280, 720))

    while True:
        a, img = cap.read()

        if not a:
            break

        dst = cv2.undistort(img, mtx, dist, None, mtx)

        dst = cv2.fastNlMeansDenoisingColored(dst, None, 3, 10, 7, 21)

        dst_orig_crp = dst[360:-1, :]

        dst_triangle = triangle_roi2(dst)

        dst_crop = dst_triangle[360:-1, :]

        pts_1 = np.array(((280, 330), (1090, 310), (820, 165), (530, 150)), dtype=np.float32)
        pts_2 = np.array(((0, 200), (50, 200), (50, 0), (0, 0)), dtype=np.float32)

        h = cv2.getPerspectiveTransform(pts_1, pts_2)

        new_img = cv2.warpPerspective(dst_crop, h, (70, 250))

        new_img_hsv = cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV)

        mask_w = cv2.inRange(new_img_hsv, (0, 0, 184), (255, 255, 255))

        mask_y = cv2.inRange(new_img_hsv, (0, 37, 15), (255, 255, 255))

        mask = cv2.bitwise_or(mask_w, mask_y)

        try:
            birdeye_equation, birdeye_lanepoints, direction = lane_estimates2(mask)
        except:
            pass

        lane_mask = np.zeros_like(new_img)

        cv2.fillPoly(lane_mask, [birdeye_lanepoints], (255, 0, 0))

        cv2.line(lane_mask, (birdeye_lanepoints[0, 0], birdeye_lanepoints[0, 1]),
                 (birdeye_lanepoints[1, 0], birdeye_lanepoints[1, 1]), (0, 0, 255), 10)

        cv2.line(lane_mask, (birdeye_lanepoints[-2, 0], birdeye_lanepoints[-2, 1]),
                 (birdeye_lanepoints[-1, 0], birdeye_lanepoints[-1, 1]), (0, 0, 255), 10)

        # cv2.imshow("mask", lane_mask)
        # cv2.waitKey(0)

        lane_img = cv2.warpPerspective(lane_mask, np.linalg.inv(h), (dst_orig_crp.shape[1], dst_orig_crp.shape[0]))

        dst_final_crop = cv2.addWeighted(dst_orig_crp, 0.7, lane_img, 0.3, 0)

        dst[360:-1, :] = dst_final_crop

        dst = cv2.putText(dst, direction, (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        # cv2.imshow("...", dst)
        # cv2.waitKey(0)
        out.write(dst)
