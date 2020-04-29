import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform


def plot_lines_between_nodes(warped_points, bird_image, d_thresh):
    p = np.array(warped_points)
    dist_condensed = pdist(p)
    dist = squareform(dist_condensed)

    # Close enough: 10 feet mark
    dd = np.where(dist < d_thresh * 6 / 10)
    close_p = []
    color_10 = (80, 172, 110)
    lineThickness = 4
    ten_feet_violations = len(np.where(dist_condensed < 10 / 6 * d_thresh)[0])
    for i in range(int(np.ceil(len(dd[0]) / 2))):
        if dd[0][i] != dd[1][i]:
            point1 = dd[0][i]
            point2 = dd[1][i]

            close_p.append([point1, point2])

            cv2.line(
                bird_image,
                (p[point1][0], p[point1][1]),
                (p[point2][0], p[point2][1]),
                color_10,
                lineThickness,
            )

    # Really close: 6 feet mark
    dd = np.where(dist < d_thresh)
    six_feet_violations = len(np.where(dist_condensed < d_thresh)[0])
    total_pairs = len(dist_condensed)
    danger_p = []
    color_6 = (52, 92, 227)
    for i in range(int(np.ceil(len(dd[0]) / 2))):
        if dd[0][i] != dd[1][i]:
            point1 = dd[0][i]
            point2 = dd[1][i]

            danger_p.append([point1, point2])
            cv2.line(
                bird_image,
                (p[point1][0], p[point1][1]),
                (p[point2][0], p[point2][1]),
                color_6,
                lineThickness,
            )
    # Display Birdeye view
    cv2.imshow("Bird Eye View", bird_image)
    cv2.waitKey(1)

    return six_feet_violations, ten_feet_violations, total_pairs


def plot_points_on_bird_eye_view(frame, pedestrian_boxes, M, scale_w, scale_h):
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]

    node_radius = 10
    color_node = (192, 133, 156)
    thickness_node = 20
    solid_back_color = (41, 41, 41)

    blank_image = np.zeros(
        (int(frame_h * scale_h), int(frame_w * scale_w), 3), np.uint8
    )
    blank_image[:] = solid_back_color
    warped_pts = []
    for i in range(len(pedestrian_boxes)):

        mid_point_x = int(
            (pedestrian_boxes[i][1] * frame_w + pedestrian_boxes[i][3] * frame_w) / 2
        )
        mid_point_y = int(
            (pedestrian_boxes[i][0] * frame_h + pedestrian_boxes[i][2] * frame_h) / 2
        )

        pts = np.array([[[mid_point_x, mid_point_y]]], dtype="float32")
        warped_pt = cv2.perspectiveTransform(pts, M)[0][0]
        warped_pt_scaled = [int(warped_pt[0] * scale_w), int(warped_pt[1] * scale_h)]

        warped_pts.append(warped_pt_scaled)
        bird_image = cv2.circle(
            blank_image,
            (warped_pt_scaled[0], warped_pt_scaled[1]),
            node_radius,
            color_node,
            thickness_node,
        )

    return warped_pts, bird_image


def get_camera_perspective(img, src_points):
    IMAGE_H = img.shape[0]
    IMAGE_W = img.shape[1]
    src = np.float32(np.array(src_points))
    dst = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    return M, M_inv


def put_text(frame, text, text_offset_y=25):
    font_scale = 0.8
    font = cv2.FONT_HERSHEY_SIMPLEX
    rectangle_bgr = (35, 35, 35)
    (text_width, text_height) = cv2.getTextSize(
        text, font, fontScale=font_scale, thickness=1
    )[0]
    # set the text start position
    text_offset_x = frame.shape[1] - 400
    # make the coords of the box with a small padding of two pixels
    box_coords = (
        (text_offset_x, text_offset_y + 5),
        (text_offset_x + text_width + 2, text_offset_y - text_height - 2),
    )
    frame = cv2.rectangle(
        frame, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED
    )
    frame = cv2.putText(
        frame,
        text,
        (text_offset_x, text_offset_y),
        font,
        fontScale=font_scale,
        color=(255, 255, 255),
        thickness=1,
    )

    return frame, 2 * text_height + text_offset_y


def calculate_stay_at_home_index(total_pedestrians_detected, frame_num, fps):
    normally_people = 10
    pedestrian_per_sec = np.round(total_pedestrians_detected / frame_num, 1)
    sh_index = 1 - pedestrian_per_sec / normally_people
    return pedestrian_per_sec, sh_index


def plot_pedestrian_boxes_on_image(frame, pedestrian_boxes):
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    thickness = 2
    # color_node = (192, 133, 156)
    color_node = (160, 48, 112)
    # color_10 = (80, 172, 110)

    for i in range(len(pedestrian_boxes)):
        pt1 = (
            int(pedestrian_boxes[i][1] * frame_w),
            int(pedestrian_boxes[i][0] * frame_h),
        )
        pt2 = (
            int(pedestrian_boxes[i][3] * frame_w),
            int(pedestrian_boxes[i][2] * frame_h),
        )

        frame_with_boxes = cv2.rectangle(frame, pt1, pt2, color_node, thickness)


    return frame_with_boxes
