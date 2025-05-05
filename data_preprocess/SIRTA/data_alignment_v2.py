'''
# This code is used for aligning images with time series data.
# In addition, image distortion correction is performed during the alignment process.
'''
import cv2
import numpy as np
import math
import time
import os
import glob
from datetime import datetime, timedelta
import logging
import warnings
import argparse
from datetime import datetime
from typing import List, Tuple
import h5py


def parser_txt_(txt_path, valid_time_list):
    parsed_data = []
    unvalid_time_list = []
    with open(txt_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            line = line.strip()
            if not line:
                continue

            columns = line.split()
                
            try:
                time_str = columns[0]
                col2 = float(columns[1])
                col3 = float(columns[2])
                col4 = float(columns[3])
                col20 = float(columns[19])
                col21 = float(columns[20])
                col22 = float(columns[21])
                
                time_str = time_str.strip()
                time_tmp = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
                time_str = time_tmp.strftime("%Y%m%d%H%M%S")
                
                if time_str not in valid_time_list:
                    continue
                if np.isnan(col2) or np.isnan(col3) or np.isnan(col4) or np.isnan(col20) or np.isnan(col21) or np.isnan(col22):
                    time_str_index = valid_time_list.index(time_str)
                    unvalid_time_list.append(time_str_index)
                    continue
                parsed_data.append({
                    'DateTime': time_str,
                    'Direct Solar Flux (W/m2)': col2,
                    'Diffuse Solar Flux (W/m2)': col3,
                    'Global Solar Flux (W/m2)': col4,
                    'Solar Zenith Angle (deg)': col20,
                    'Solar Azimuthal Angle (deg)': col21,
                    'precipitation zone 2 (mm/min)': col22
                })
            except (ValueError, IndexError) as e:
                print(f"parser txt error: {line}")
                continue
                
    return parsed_data, unvalid_time_list


def cut(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (_, thresh) = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    x,y,w,h = cv2.boundingRect(cnts)
    r = max(w/ 2, h/ 2)

    '''！！！这里改裁剪的参数！！！！'''
    r = r

    img_valid = img[y:y+h, x:x+w]
    return img_valid, int(r)


def undistort(src,r):
    R = 2*r
    Pi = np.pi
    dst = np.zeros((R, R, 3)).astype(np.uint8)
    src_h, src_w, _ = src.shape
    x0, y0 = src_w//2, src_h//2

    for dst_y in range(0, R):

        theta =  Pi - (Pi/R)*dst_y
        temp_theta = math.tan(theta)**2

        for dst_x in range(0, R):

            phi = Pi - (Pi/R)*dst_x
            temp_phi = math.tan(phi)**2

            tempu = r/(temp_phi+ 1 + temp_phi/temp_theta)**0.5
            tempv = r/(temp_theta + 1 + temp_theta/temp_phi)**0.5

            if (phi < Pi/2):
                u = x0 + tempu
            else:
                u = x0 - tempu

            if (theta < Pi/2):
                v = y0 + tempv
            else:
                v = y0 - tempv

            if (u>=0 and v>=0 and u+0.5<src_w and v+0.5<src_h):
                dst[dst_y, dst_x, :] = src[int(v+0.5)][int(u+0.5)]

                # src_x, src_y = u, v
                # src_x_0 = int(src_x)
                # src_y_0 = int(src_y)
                # src_x_1 = min(src_x_0 + 1, src_w - 1)
                # src_y_1 = min(src_y_0 + 1, src_h - 1)
                #
                # value0 = (src_x_1 - src_x) * src[src_y_0, src_x_0, :] + (src_x - src_x_0) * src[src_y_0, src_x_1, :]
                # value1 = (src_x_1 - src_x) * src[src_y_1, src_x_0, :] + (src_x - src_x_0) * src[src_y_1, src_x_1, :]
                # dst[dst_y, dst_x, :] = ((src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1 + 0.5).astype('uint8')

    return dst



def align_and_merge_data(txt_file_list, image_folder_list, reshape_size=64, distort=True):
    all_time_list = []
    all_direct_list = []
    all_diffuse_list = []
    all_global_list = []
    all_solar_zenith_list = []
    all_solar_azimuth_list = []
    all_precipitation_list = []
    all_image_data_list = []

    for i, image_folder in enumerate(image_folder_list):
        image_file_list = glob.glob(os.path.join(image_folder, "*.jpg"))
        image_file_list = sorted(image_file_list)
        image_filename_list = [((os.path.basename(image_file)).split(".")[0]).split("_")[0] for image_file in image_file_list]

        txt_file = txt_file_list[i]
        parsed_data, unvalid_time_list = parser_txt_(txt_file, image_filename_list)
        image_file_list = [image_file_list[i] for i in range(len(image_file_list)) if i not in unvalid_time_list]
        image_filename_list = [image_filename_list[i] for i in range(len(image_filename_list)) if i not in unvalid_time_list] # 工具数组，后面没用

        time_list = [parsed_data[i]['DateTime'] for i in range(len(parsed_data))]

        for i in range(len(image_filename_list)):
            if image_filename_list[i] not in time_list:
                image_file_list.pop(i)
        
        if len(parsed_data) != len(image_file_list):
            print(txt_file, "length dismatch")
            continue

        image_data_list_tmp = [np.zeros((reshape_size, reshape_size, 3), dtype=np.float32) for _ in range(len(image_file_list))]
        for j, tmp_file in enumerate(image_file_list):
            tmp_img = cv2.imread(tmp_file)
            if distort:
                tmp_img = undistort(tmp_img, tmp_img.shape[0]//2)

            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
            tmp_img_tmp = cv2.resize(tmp_img, (reshape_size, reshape_size), interpolation=cv2.INTER_AREA)
            tmp_img_tmp = tmp_img_tmp.astype(np.float32)
            image_data_list_tmp[j] = tmp_img_tmp
        

        direct_list = [parsed_data[i]['Direct Solar Flux (W/m2)'] for i in range(len(parsed_data))]
        diffuse_list = [parsed_data[i]['Diffuse Solar Flux (W/m2)'] for i in range(len(parsed_data))]
        global_list = [parsed_data[i]['Global Solar Flux (W/m2)'] for i in range(len(parsed_data))]
        solar_zenith_list = [parsed_data[i]['Solar Zenith Angle (deg)'] for i in range(len(parsed_data))]
        solar_azimuth_list = [parsed_data[i]['Solar Azimuthal Angle (deg)'] for i in range(len(parsed_data))]
        precipitation_list = [parsed_data[i]['precipitation zone 2 (mm/min)'] for i in range(len(parsed_data))]
        wind_list = []
        
        all_time_list.append(time_list)
        all_direct_list.append(direct_list)
        all_diffuse_list.append(diffuse_list)
        all_global_list.append(global_list)
        all_solar_zenith_list.append(solar_zenith_list)
        all_solar_azimuth_list.append(solar_azimuth_list)
        all_precipitation_list.append(precipitation_list)
        all_image_data_list.append(image_data_list_tmp)

        print("finish {}".format(image_folder))
    
    all_time_list = [item for sublist in all_time_list for item in sublist]
    all_direct_list = [item for sublist in all_direct_list for item in sublist]
    all_diffuse_list = [item for sublist in all_diffuse_list for item in sublist]
    all_global_list = [item for sublist in all_global_list for item in sublist]
    all_solar_zenith_list = [item for sublist in all_solar_zenith_list for item in sublist]
    all_solar_azimuth_list = [item for sublist in all_solar_azimuth_list for item in sublist]
    all_precipitation_list = [item for sublist in all_precipitation_list for item in sublist]
    all_image_data_list = [item for sublist in all_image_data_list for item in sublist]

    all_time_list = [datetime.strptime(time_str, "%Y%m%d%H%M%S") for time_str in all_time_list]
    all_time_list = np.array(all_time_list)
    all_direct_list = np.array(all_direct_list)
    all_diffuse_list = np.array(all_diffuse_list)
    all_global_list = np.array(all_global_list)
    all_solar_zenith_list = np.array(all_solar_zenith_list)
    all_solar_azimuth_list = np.array(all_solar_azimuth_list)
    all_precipitation_list = np.array(all_precipitation_list)
    all_image_data_list = np.array(all_image_data_list)


    return all_time_list, all_direct_list, all_diffuse_list, all_global_list, all_solar_zenith_list, all_solar_azimuth_list, all_precipitation_list, all_image_data_list



date_start = "20170101"
date_end = "20191231"
base_dir = "sirta\\sky_image\\data_cut_selected"
start_date = datetime.strptime(date_start, "%Y%m%d")
end_date = datetime.strptime(date_end, "%Y%m%d")
date_list = []
current_date = start_date
while current_date <= end_date:
    date_list.append(current_date)
    current_date += timedelta(days=1)
for date in date_list:
    date_str = date.strftime("%Y%m%d")
    if not os.path.exists(os.path.join(base_dir, date_str)):
        os.makedirs(os.path.join(base_dir, date_str))

pattern = "sirta\\irradiance\\*\\*\\*\\radflux_1b_Lz2M1minIsolys2PrayDp-QC_v01_*.txt"
txt_file_list = glob.glob(pattern)
txt_file_list = sorted(txt_file_list)

pattern = "sirta\\sky_image\\data_cut_selected\\*"
image_folder_list = glob.glob(pattern)
image_folder_list = sorted(image_folder_list)
if len(image_folder_list) != len(txt_file_list):
    raise ValueError("image_folder_list and txt_file_list have different length")

distort = True

all_time_arr, all_direct_arr, all_diffuse_arr, all_global_arr, all_solar_zenith_arr, all_solar_azimuth_arr, all_precipitation_arr, all_image_data_arr = align_and_merge_data(txt_file_list, image_folder_list,reshape_size=64, distort=distort)


np.save("sirta_10min_timestamp.npy", all_time_arr)

if distort:
    with h5py.File("sirta_cut_64_10min_dc.hdf5", "w") as f:
        f.create_dataset("Direct Solar Flux (W/m2)", data=all_direct_arr)
        f.create_dataset("Diffuse Solar Flux (W/m2)", data=all_diffuse_arr)
        f.create_dataset("Global Solar Flux (W/m2)", data=all_global_arr)
        f.create_dataset("Solar Zenith Angle (deg)", data=all_solar_zenith_arr)
        f.create_dataset("Solar Azimuthal Angle (deg)", data=all_solar_azimuth_arr)
        f.create_dataset("precipitation zone 2 (mm/min)", data=all_precipitation_arr)
        f.create_dataset("sky_images", data=all_image_data_arr)
else:
    with h5py.File("sirta_cut_64_10min_udc.hdf5", "w") as f:
        f.create_dataset("Direct Solar Flux (W/m2)", data=all_direct_arr)
        f.create_dataset("Diffuse Solar Flux (W/m2)", data=all_diffuse_arr)
        f.create_dataset("Global Solar Flux (W/m2)", data=all_global_arr)
        f.create_dataset("Solar Zenith Angle (deg)", data=all_solar_zenith_arr)
        f.create_dataset("Solar Azimuthal Angle (deg)", data=all_solar_azimuth_arr)
        f.create_dataset("precipitation zone 2 (mm/min)", data=all_precipitation_arr)
        f.create_dataset("sky_images", data=all_image_data_arr)




