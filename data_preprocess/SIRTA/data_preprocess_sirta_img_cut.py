'''
# Since the center point of images in the SIRTA dataset varies, 
# an algorithm is used here to crop the effective region of each image.

# For each file in the daily list (with 10-minute resolution):
#   1. Calculate its effective region using find_effective_region, obtaining a list of radii and center points.
#   2. Check if there are any radii in the range of 348-352. If so, filter out these images and calculate the variance of their center point coordinates.
#      - If the variance is within an acceptable range, use the mean of these center points as the valid center point.
#      - Use a radius of 350 and the computed center point as the effective center, and recalculate the effective region for each image in the list.
#   3. If no radii fall within 348-352, log the date and use the previous day's center point as the valid center point, then recalculate the effective region for each image in the list.
'''
import cv2
import math
import time
import os
import glob
import logging
import warnings
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple

def find_effective_region(img, threshold=18):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    x1, y1, x2, y2 = img.shape[1], img.shape[0], 0, 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x1, y1 = min(x1, x), min(y1, y)
        x2, y2 = max(x2, x + w), max(y2, y + h)
    
    side = max(x2 - x1, y2 - y1)
    center_x, center_y = (x1 + x2) // 2 , (y1 + y2) // 2
    
    new_x1, new_y1 = max(0, center_x - side // 2), max(0, center_y - side // 2)
    new_x2, new_y2 = min(img.shape[1], center_x + side // 2), min(img.shape[0], center_y + side // 2)
    
    r = side // 2
    img_valid = img[new_y1:new_y2, new_x1:new_x2]

    return img_valid, r, center_x, center_y


def find_effective_region_prior(img, center_x, center_y, r):
    new_x1, new_y1 = max(0, center_x - r), max(0, center_y - r)
    new_x2, new_y2 = min(img.shape[1], center_x + r), min(img.shape[0], center_y + r)
    
    img_valid = img[new_y1:new_y2, new_x1:new_x2]
    
    return img_valid


def down_sample(img, resize=256):
    img_down = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_AREA)
    return img_down


def crop_circular_region(img, r, r_crop=5):
    center_x, center_y = r, r
    r_new = r - r_crop
    mask = np.zeros_like(img)
    cv2.circle(mask, (center_x, center_y), r_new, (255, 255, 255), -1)
    img = cv2.bitwise_and(img, mask)

    img_valid_cut = img[center_y - r_new:center_y + r_new, center_x - r_new:center_x + r_new]

    return img_valid_cut, r_new


def round_up_to_ten_minutes(time_str):
    time_obj = datetime.strptime(time_str, "%H%M%S")
    
    minutes = time_obj.minute
    remainder = minutes % 10
    if remainder == 0:
        return time_str
    else:
        add_minutes = 10 - remainder
        rounded_time = time_obj + timedelta(minutes=add_minutes)
    
    if rounded_time.hour >= 24:
        rounded_time = rounded_time.replace(hour=0, minute=0, second=0)
    
    return rounded_time.strftime("%H%M%S")


def get_daily_image_paths(root_path, time_horizon=10):

    # pattern = os.path.join(root_path, "*", "*", "*", "srf02_*", "*")
    pattern = os.path.join(root_path, "*", "*", "srf02_*", "*")
    # pattern = os.path.join(root_path, "*", "srf02_*", "*")
    # pattern = os.path.join(root_path, "srf02_*", "*")
    all_paths = glob.glob(pattern)
    print("len(all_paths): ", len(all_paths))
    all_image_paths = [[] for _ in range(len(all_paths))]
    for i, tmp_path in enumerate(all_paths):
        parts = tmp_path.split(os.sep)
        # print(parts)
        file_date = parts[-1]
        file_num = int(parts[-2].split("_")[-1])
        cal_file_num = file_num//(time_horizon//2) + time_horizon
        today_start_time_str = parts[-2].split("_")[-2]
        today_start_time_str_round = round_up_to_ten_minutes(today_start_time_str)
        
        for j in range(0, cal_file_num):
            tmp_time = datetime.strptime(today_start_time_str_round, "%H%M%S") + timedelta(minutes=j*time_horizon)
            tmp_time_str = tmp_time.strftime("%H%M%S")
            tmp_file_path = os.path.join(tmp_path, file_date+tmp_time_str+"_01.jpg")
            if os.path.exists(tmp_file_path):
                all_image_paths[i].append(tmp_file_path)


    return all_paths, all_image_paths


def cal_effective_region(all_paths, all_image_paths, root_path="sirta\\sky_image\\data_cut", resize_l=256):
    lst_center_x = -1
    lst_center_y = -1
    mean_center_x = -1
    mean_center_y = -1
    r = 350
    resize_l = 256

    for i, tmp_path in enumerate(all_paths):
        
        center_x_list = [0 for _ in range(len(all_image_paths[i]))]
        center_y_list = [0 for _ in range(len(all_image_paths[i]))]
        r_list = [0 for _ in range(len(all_image_paths[i]))]

        parts = tmp_path.split(os.sep)
        file_date = parts[-1]
        tgt_path = root_path + os.sep + file_date
        if not os.path.exists(tgt_path):
            os.makedirs(tgt_path)

        print("processing:",file_date)
        cur_time = time.time()
        
        # scan effective region 
        for j, tmp_file_path in enumerate(all_image_paths[i]):
            img = cv2.imread(tmp_file_path)
            _, r, center_x, center_y = find_effective_region(img)
            center_x_list[j] = center_x
            center_y_list[j] = center_y
            r_list[j] = r
        

        # filter valid region
        center_x_list_valid = []
        center_y_list_valid = []
        flg_valid = 1
        for j in range(len(r_list)):
            if r_list[j] >= 346 and r_list[j] <= 354:
                center_x_list_valid.append(center_x_list[j])
                center_y_list_valid.append(center_y_list[j])
        
        # calculate mean and var
        if len(center_x_list_valid) > 0:
            mean_center_x = np.mean(center_x_list_valid)
            mean_center_y = np.mean(center_y_list_valid)
            var_center_x = np.var(center_x_list_valid)+0.000001
            var_center_y = np.var(center_y_list_valid)+0.000001
            # print("var_center_x: ", var_center_x)
            # print("var_center_y: ", var_center_y)
            if var_center_x > 25 and var_center_y > 25:
                flg_valid = 0
                logging.info(f"{tmp_path}: var_center_x={var_center_x}, var_center_y={var_center_y}")                
        else:
            flg_valid = 0
            logging.info(f"{tmp_path}: no valid center, lst_center_x={lst_center_x}, lst_center_y={lst_center_y}")
        
        # crop effective region
        r = 350
        if flg_valid == 1:
            for j, tmp_file_path in enumerate(all_image_paths[i]):
                # print("processing:",tmp_file_path)

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    img = cv2.imread(tmp_file_path)

                    if w and any("Corrupt JPEG data" in str(warn.message) for warn in w):
                        logging.info(f"{tmp_file_path.split(os.sep)[-1]}: Corrupt JPEG data")
                        continue

                img_valid = find_effective_region_prior(img, int(mean_center_x), int(mean_center_y), r)
                img_valid = down_sample(img_valid, resize=resize_l)
                img_valid, _ = crop_circular_region(img_valid, resize_l//2, r_crop=10)
                img_valid = img_valid.astype(np.uint8)
                if not os.path.exists(tgt_path + os.sep + tmp_file_path.split(os.sep)[-1]):
                    cv2.imwrite(tgt_path + os.sep + tmp_file_path.split(os.sep)[-1], img_valid)
                    logging.info("save image to {}".format(tgt_path + os.sep + tmp_file_path.split(os.sep)[-1]))

            lst_center_x = mean_center_x
            lst_center_y = mean_center_y    
        else:
            for j, tmp_file_path in enumerate(all_image_paths[i]):
                img = cv2.imread(tmp_file_path)
                img_valid = find_effective_region_prior(img, int(lst_center_x), int(lst_center_y), r)
                img_valid = down_sample(img_valid, resize=resize_l)
                img_valid, _  = crop_circular_region(img_valid, resize_l//2, r_crop=10)
                img_valid = img_valid.astype(np.uint8)
                if not os.path.exists(tgt_path + os.sep + tmp_file_path.split(os.sep)[-1]):
                    cv2.imwrite(tgt_path + os.sep + tmp_file_path.split(os.sep)[-1], img_valid)
                    logging.info("save image to {}".format(tgt_path + os.sep + tmp_file_path.split(os.sep)[-1]))
        
        print("time cost for {} : {}".format(file_date, time.time() - cur_time))
        logging.info("time cost for {} : {}".format(file_date, time.time() - cur_time))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="2017")
    parser.add_argument("--resize_l", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=10)

    args = parser.parse_args()
    args.log_file = "log_" + args.root_path + ".txt"
    args.root_path = "sirta\\sky_image\\" + args.root_path

    logging.basicConfig(filename=args.log_file,level=logging.INFO, format='%(message)s')

    all_paths, all_image_paths = get_daily_image_paths(args.root_path, time_horizon=args.horizon)
    cal_effective_region(all_paths, all_image_paths, resize_l=args.resize_l)



