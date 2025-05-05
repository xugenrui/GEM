'''
In the 2017 SIRTA dataset, some overexposed images may exist for certain periods due to equipment issues.
The following data cleaning code is used to detect and remove these overexposed images.
Image processing algorithms are applied for detection, and their effectiveness has been validated in our tests.
'''

import cv2
import numpy as np
import os
from typing import List
import logging
import glob


def histogram_diff(img1, img2):
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return diff


def brightness_stats(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    return mean_brightness, std_brightness


def is_overexposed(prev_img, curr_img, next_img, hist_thresh=0.3, brightness_thresh=20, prev_img_2=None, next_img_2=None):
    hist_diff_prev = histogram_diff(curr_img, prev_img)
    hist_diff_next = histogram_diff(curr_img, next_img)
    if prev_img_2 is not None and next_img_2 is not None:
        hist_diff_prev_2 = histogram_diff(curr_img, prev_img_2)
        hist_diff_next_2 = histogram_diff(curr_img, next_img_2)

    logging.info("hist_diff_prev_{},hist_diff_next_{}".format(hist_diff_prev,hist_diff_next))
    
    mean_curr, std_curr = brightness_stats(curr_img)
    mean_prev, std_prev = brightness_stats(prev_img)
    mean_next, std_next = brightness_stats(next_img)
    if prev_img_2 is not None and next_img_2 is not None:
        mean_prev_2, std_prev_2 = brightness_stats(prev_img_2)
        mean_next_2, std_next_2 = brightness_stats(next_img_2)

    logging.info(f"mean_prev_{mean_prev},mean_curr_{mean_curr},mean_next_{mean_next}")


    hist_condition = hist_diff_prev > hist_thresh and hist_diff_next > hist_thresh
    brightness_condition = (mean_curr - mean_prev) > brightness_thresh and (mean_curr - mean_next) > brightness_thresh
    if prev_img_2 is not None and next_img_2 is not None:
        hist_condition2 = hist_diff_prev_2 > hist_thresh and hist_diff_next_2 > hist_thresh
        brightness_condition2 = (mean_curr - mean_prev_2) > brightness_thresh and (mean_curr - mean_next) > brightness_thresh
        hist_condition3 = hist_diff_prev > hist_thresh and hist_diff_next_2 > hist_thresh
        brightness_condition3 = (mean_curr - mean_prev) > brightness_thresh and (mean_curr - mean_next_2) > brightness_thresh

    
    # logging.info(f"hist_condition_{hist_condition},brightness_condition_{brightness_condition},saturation_condition_{saturation_condition},saturation_condition2_{saturation_condition2}")
    # if hist_condition and brightness_condition and (saturation_condition or saturation_condition2):
    if hist_condition and brightness_condition:
        return True
    elif prev_img_2 is not None and next_img_2 is not None:
        if (hist_condition2 and brightness_condition2) or (hist_condition3 and brightness_condition3):
            return True
        else:
            return False
    else:
        return False

# hist_diff_mean_0.18795436602230864,brightness_diff_mean_5.545137918089907,sat_diff_mean_6.78640550937696
# hist_diff_std_0.12929867975403028,brightness_diff_std_9.158991599817199,sat_diff_std_9.706124551848404
def detect_overexposed_images(image_paths: List[str]):
    overexposed_images = []
    for i in range(1, len(image_paths) - 1):
        logging.info("image_paths_{}".format(image_paths[i]))
        prev_img = cv2.imread(image_paths[i - 1])
        curr_img = cv2.imread(image_paths[i])
        next_img = cv2.imread(image_paths[i + 1])
        if i > 1:
            prev_img_2 = cv2.imread(image_paths[i - 2])
        else:
            prev_img_2 = None
        if i < len(image_paths) - 2:
            next_img_2 = cv2.imread(image_paths[i + 2])
        else:
            next_img_2 = None

        if prev_img is None or curr_img is None or next_img is None:
            logging.warning(f"read image failed: {image_paths[i]}")
            continue

        if is_overexposed(prev_img, curr_img, next_img, prev_img_2=prev_img_2, next_img_2=next_img_2):
            logging.info(f"detect overexposed image: {image_paths[i]}")
            overexposed_images.append(image_paths[i])

    return overexposed_images


def stat_overexposed_images(image_paths: List[str]):
    overexposed_images = []
    hist_diff_list = []
    brightness_diff_list = []
    sat_diff_list = []
    sat_diff_ratio_list = []
    for i in range(1, len(image_paths) - 1):
        logging.info("image_paths_{}".format(image_paths[i]))
        prev_img = cv2.imread(image_paths[i - 1])
        curr_img = cv2.imread(image_paths[i])
        next_img = cv2.imread(image_paths[i + 1])

        hist_diff, brightness_diff, sat_diff, sat_diff_ratio = is_overexposed(prev_img, curr_img, next_img)
        hist_diff_list.append(hist_diff)
        brightness_diff_list.append(brightness_diff)
        sat_diff_list.append(sat_diff)
        sat_diff_ratio_list.append(sat_diff_ratio)

    return hist_diff_list, brightness_diff_list, sat_diff_list, sat_diff_ratio_list


def get_exposure_images_from_txt(txt_path,tgt_path):
    with open(txt_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('detect'):
                file_path = line.split(": ")[-1]
                file_path = file_path.strip()
                file_name = file_path.split("\\")[-1]
                img = cv2.imread(file_path)
                if img is None:
                    logging.warning(f"read image failed: {file_path}")
                    continue
                cv2.imwrite(os.path.join(tgt_path,file_name),img)


def delete_exposure_images_from_folder(folder_path,tgt_path):
    file_list = os.listdir(folder_path)
    for file in file_list:
        if file.endswith(".jpg"):
            file_name = file.split(os.sep)[-1]
            date_str = ((file_name).split(".")[0])[0:8]
            os.remove(os.path.join(tgt_path,date_str,file_name))


def delete_night_images(file_list,base_light=40):
    for file in file_list:
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness < base_light:
            logging.info("delete morning image: {}".format(file))
            os.remove(file)
        else:
            break
    file_list_reverse = file_list[::-1]
    for file in file_list_reverse:
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness < base_light:
            logging.info("delete night image: {}".format(file))
            os.remove(file)
        else:
            break


if __name__ == "__main__":

    logging.basicConfig(filename='night_images.log', level=logging.INFO, format='%(message)s')
    pattern = "sirta\\sky_image\\data_cut_selected\\*"
    image_folders = glob.glob(pattern)
    image_folders = sorted(image_folders)
    for image_folder in image_folders:
        image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".jpg")])
        print("processing image_folder_{}".format(image_folder))
        delete_night_images(image_files)

    
