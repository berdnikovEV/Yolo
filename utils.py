"""Contains utility functions for Yolo v3 model."""
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from seaborn import color_palette
import cv2
from typing import List

CONFIG = json.load(open('config.json'))

def get_dist(old_point, new_point):
    return ((old_point[0] - new_point[0])**2 + (old_point[1] - new_point[1])**2)**0.5


def add_new_point(point_list:List[List], new_point, vehicle_type):
    # point_list = [[old],[upd]]
    if not point_list[0]:
        point_list[1].append([vehicle_type, [None, None], new_point])
        return vehicle_type

    last_point_list = [path[-1] for path in point_list[0]]

    distance_array = [get_dist(last_point, new_point) for last_point in last_point_list]

    index = distance_array.index(min(distance_array))

    if distance_array[index] < 50:
        point_list[1].append(point_list[0][index])
        point_list[1][-1].append(new_point)

        # point_list[0].remove(point_list[0][index])
        del point_list[0][index]
        vehicle_type = point_list[1][-1][0]

    else:
        point_list[1].append([vehicle_type, [None, None], new_point])

    return vehicle_type


def load_images(img_names, model_size):
    """Loads images in a 4D array.
    Args:
        img_names: A list of images names.
        model_size: The input size of the model.
    Returns:
        A 4D NumPy array.
    """
    imgs = []

    for img_name in img_names:
        img = Image.open(img_name)
        img = img.resize(size=model_size)
        img = np.array(img, dtype=np.float32)
        img = np.expand_dims(img[:, :, :3], axis=0)
        imgs.append(img)

    imgs = np.concatenate(imgs)

    return imgs


def load_class_names(file_name):
    """Returns a list of class names read from `file_name`."""
    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names


def draw_boxes(img_names, boxes_dicts, class_names, model_size):
    """Draws detected boxes.
    Args:
        img_names: A list of input images names.
        boxes_dicts: A class-to-boxes dictionary.
        class_names: A class names list.
        model_size: The input size of the model.
    Returns:
        None.
    """
    colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)
    for num, img_name, boxes_dict in zip(range(len(img_names)), img_names,
                                         boxes_dicts):
        img = Image.open(img_name)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font='./data/fonts/futur.ttf',
                                  size=(img.size[0] + img.size[1]) // 100)
        resize_factor = \
            (img.size[0] / model_size[0], img.size[1] / model_size[1])
        for cls in range(len(class_names)):
            boxes = boxes_dict[cls]
            if np.size(boxes) != 0:
                color = colors[cls]
                for box in boxes:
                    xy, confidence = box[:4], box[4]
                    xy = [xy[i] * resize_factor[i % 2] for i in range(4)]
                    x0, y0 = xy[0], xy[1]
                    thickness = (img.size[0] + img.size[1]) // 200
                    for t in np.linspace(0, 1, thickness):
                        xy[0], xy[1] = xy[0] + t, xy[1] + t
                        xy[2], xy[3] = xy[2] - t, xy[3] - t
                        draw.rectangle(xy, outline=tuple(color))
                    text = '{} {:.1f}%'.format(class_names[cls],
                                               confidence * 100)
                    text_size = draw.textsize(text, font=font)
                    draw.rectangle(
                        [x0, y0 - text_size[1], x0 + text_size[0], y0],
                        fill=tuple(color))
                    draw.text((x0, y0 - text_size[1]), text, fill='black',
                              font=font)
                    print('{} {:.2f}%'.format(class_names[cls],
                                              confidence * 100))

        rgb_img = img.convert('RGB')

        rgb_img.save('./detections/detection_' + str(num + 1) + '.jpg')


def draw_frame(frame, frame_size, boxes_dicts, class_names, model_size, point_list):
    """Draws detected boxes in a video frame.
    Args:
        frame: A video frame.
        frame_size: A tuple of (frame width, frame height).
        boxes_dicts: A class-to-boxes dictionary.
        class_names: A class names list.
        model_size:The input size of the model.
        point_list: array of paths taken by v
    Returns:
        None.
    """
    boxes_dict = boxes_dicts[0]
    resize_factor = (frame_size[0] / model_size[1], frame_size[1] / model_size[0])
    colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)
    for cls in range(len(class_names)):
        try:
            if class_names[cls] in ['car', 'truck', 'bus']:
                boxes = boxes_dict[cls]
                color = colors[cls]
                color = tuple([int(x) for x in color])
                if np.size(boxes) != 0:
                    for box in boxes:
                        vehicle_type = class_names[cls]
                        xy = box[:4]
                        xy = [int(xy[i] * resize_factor[i % 2]) for i in range(4)]
                        point = ((xy[0] + xy[2])/2, (xy[1] + xy[3])/2)

                        distance_list = [get_dist(point, x[-1]) for x in point_list[1]]
                        if distance_list and min(distance_list) < 30:
                            continue

                        vehicle_type = add_new_point(point_list, point, vehicle_type)

                        if point_list[1]:
                            path = point_list[1][-1]
                        else:
                            continue

                        cv2.rectangle(frame, (xy[0], xy[1]), (xy[2], xy[3]), color[::-1], 2)
                        (test_width, text_height), baseline = cv2.getTextSize(class_names[cls],
                                                                              cv2.FONT_HERSHEY_SIMPLEX,
                                                                              0.75, 1)
                        cv2.rectangle(frame, (xy[0], xy[1]),
                                      (xy[0] + test_width, xy[1] - text_height - baseline),
                                      color[::-1], thickness=cv2.FILLED)
                        cv2.putText(frame, vehicle_type, (xy[0], xy[1] - baseline),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
                        if len(path) > 1:
                            for i in range(2, len(path)-1):
                                if path[i][0] is not None:
                                    cv2.line(frame,
                                             (int(path[i][0]), int(path[i][1])),
                                             (int(path[i+1][0]), int(path[i+1][1])),
                                             color=(0, 255, 255),
                                             thickness=3)
        except Exception as e:
            print(str(e))
    borders = CONFIG['borders']

    try:
        for border_coordinate_set in borders:
            border_coordinate_set = tuple(border_coordinate_set)
            cv2.line(frame,
                     border_coordinate_set[0:2],
                     border_coordinate_set[2:4],
                     color=(0, 0, 0),
                     thickness=2)
    except Exception as fuck:
        print(str(fuck))

    cv2.line(frame,
             border_coordinate_set[0:2],
             border_coordinate_set[2:4],
             color=(0, 0, 0),
             thickness=2)

    point_list[0], point_list[1] = point_list[1] + [x+[[999999999, 999999999]] for x in point_list[0]], []
