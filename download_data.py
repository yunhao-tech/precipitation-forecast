# """Fetcher for RAMP data stored in OSF
# To adapt it for another challenge, change the CHALLENGE_NAME and upload
# public/private data as `tar.gz` archives in dedicated OSF folders named after
# the challenge.
# """
import datetime
import numpy as np
from datetime import timedelta
import h5py
import urllib.parse
import urllib.request
import json
import shutil
import os,sys
import pygmt
import PIL
import glob
from matplotlib.pyplot import imshow
import matplotlib.animation as animation
from matplotlib import cm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import xarray as xr
import rioxarray as rio
import pygmt
from PIL import Image


CHALLENGE_NAME = "precipitation_forecast"



key = 'eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6ImVmMGYxMjZlODRmNzQzYjRiMmM0YTg0MWZmODBjZmFmIiwiaCI6Im11cm11cjEyOCJ9'

def getRadarData(key, tstamp,dirloc):
        
    url = 'https://api.dataplatform.knmi.nl/open-data/v1/datasets/radar_reflectivity_composites/versions/2.0/files/RAD_NL25_PCP_NA_'+tstamp+'.h5/url'
    headers = {'Authorization': key}

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
       meta = response.read()
        
    realurl=json.loads(meta)["temporaryDownloadUrl"]
    req = urllib.request.Request(realurl)
    fname=tstamp+".hf5"
    print(fname)
    isExist = os.path.exists(dirloc)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(dirloc)
        
    with urllib.request.urlopen(req) as response:
        with open(dirloc+fname, 'wb') as location:
            shutil.copyfileobj(response, location)


def get_files_for_specific_timestamps(tstamps_list, dirloc):
    files = []
    for tstamp in tstamps_list:
        files.append(dirloc+tstamp+".hf5")
        getRadarData(key, tstamp, dirloc)
    return files

def get_data_of_n_previous_hours(hours):
    now=datetime.datetime.utcnow()
    now = now - datetime.timedelta(hours=hours, minutes=5)
    now -= datetime.timedelta(minutes=now.minute%5)

    now.strftime("%Y%m%d%H%M")
    files = []
    start=now
    for n in range(0,hours*12): # data avaialble every 5 minutes, so 12 times per hour
        tstamp=start.strftime("%Y%m%d%H%M")
        files.append(tstamp+".hf5")
        getRadarData(key, tstamp)
        start += datetime.timedelta(minutes=5)
    return files

def morning_filenames_for_day(year, month, day):
    prefix = year+month+day
    return     [prefix+'0805',prefix+'0810',prefix+'0815',prefix+'0820',prefix+'0825',prefix+'0830',
                prefix+'0835',prefix+'0840',prefix+'0845',prefix+'0850',prefix+'0855',prefix+'0900',
                prefix+'0905',prefix+'0910',prefix+'0915',prefix+'0920',prefix+'0925',prefix+'0930',
                prefix+'0935',prefix+'0940',prefix+'0945',prefix+'0950',prefix+'0955',prefix+'1000',
                prefix+'1005',prefix+'1010',prefix+'1015',prefix+'1020',prefix+'1025',prefix+'1030',
                prefix+'1035',prefix+'1040',prefix+'1045',prefix+'1050',prefix+'1055',prefix+'1100']

def afternoon_filenames_for_day(year, month, day):
    prefix = year+month+day
    return     [prefix+'2005',prefix+'2010',prefix+'2015',prefix+'2020',prefix+'2025',prefix+'2030',
                prefix+'2035',prefix+'2040',prefix+'2045',prefix+'2050',prefix+'2055',prefix+'2100',
                prefix+'2105',prefix+'2110',prefix+'2115',prefix+'2120',prefix+'2125',prefix+'2130',
                prefix+'2135',prefix+'2140',prefix+'2145',prefix+'2150',prefix+'2155',prefix+'2200',
                prefix+'2205',prefix+'2210',prefix+'2215',prefix+'2220',prefix+'2225',prefix+'2230',
                prefix+'2235',prefix+'2240',prefix+'2245',prefix+'2250',prefix+'2255',prefix+'2300']


def raw_to_images(Private = True, Train = True):
    fig, ax = plt.subplots()

    if Train == True and Private == True:
        dirloc = './data/private_train_raw/'
    elif Train == True and Private == False:
        dirloc = './data/public_train_raw/'
    elif Train == False and Private == False:
        dirloc = './data/public_test_raw/'
    elif Train == False and Private == True:
        dirloc = './data/private_test_raw/'

    for filename in glob.iglob(dirloc + '**/*.hf5', recursive=True):
        # open raw radar data
        ax.clear()
        img = h5py.File(filename)
        original_image = np.array(img["image1"]["image_data"]).astype(float)
        cmap=np.array(img["visualisation1"]["color_palette"])
        knmimap=ListedColormap(cmap/256.0)

        # original_image[original_image == 255.0] = np.nan
        # original_image[original_image == 0.0] = np.nan
        masked_image = np.ma.array(original_image, mask=np.isnan(original_image))

        my_cmap = plt.cm.get_cmap('viridis')
        my_cmap.set_bad('white', 0)
        if Train == True and Private == True:
            day = filename[25:40]
            path_to_image_day = './data/train/private/'+day
        elif Train == True and Private == False:
            day = filename[24:39]
            path_to_image_day = './data/train/public/'+day
        elif Train == False and Private == False:
            day = filename[23:38]
            path_to_image_day = './data/test/public/'+day
        elif Train == False and Private == True:
            day = filename[24:39]
            path_to_image_day = './data/test/private/'+day
        raw_name = filename[-16:].replace('.hf5','.png')
        if not os.path.exists(path_to_image_day):
            os.makedirs(path_to_image_day)
        full_path = path_to_image_day + '/' + raw_name
        plt.imsave(full_path, masked_image, cmap=my_cmap)
        print("saved " + full_path)

def resize_images():
    for filename in glob.iglob('./data/images/' + '**/*.png', recursive=True):
        print("resizing " + filename)
        im = Image.open(filename)
        #im1 = im.crop((left, top, right, bottom))
        im1 = im.resize((70,77))
        im1.save(filename,"PNG")

def download_data_to_image(begin = datetime.date(2021,1,1), end = datetime.date(2021,1,2),private = True, train = True):
    for i in range((end - begin).days+1):
        today = begin + datetime.timedelta(days=i)
        year = str(today.year)
        month = str(today.month).zfill(2)
        day = str(today.day).zfill(2)
        tstamps_list = afternoon_filenames_for_day(year,month,day)
        if train == True and private == True:
            dirloc = './data/private_train_raw/' + today.strftime('%Y-%m-%d') + '-2005/'
        elif train == True and private == False:
            dirloc = './data/public_train_raw/' + today.strftime('%Y-%m-%d') + '-2005/'
        elif train == False and private == False:
            dirloc = './data/public_test_raw/' + today.strftime('%Y-%m-%d') + '-2005/'
        elif train == False and private == True:
            dirloc = './data/private_test_raw/' + today.strftime('%Y-%m-%d') + '-2005/'
        files = get_files_for_specific_timestamps(tstamps_list, dirloc)
    
    raw_to_images(Private = private, Train = train)


if __name__ == "__main__":

    begin = datetime.date(2021,1,1)
    end = datetime.date(2021,1,1)
    download_data_to_image(begin,end,True,True)
    begin = datetime.date(2021,1,2)
    end = datetime.date(2021,1,2)
    download_data_to_image(begin,end,True,False)
    begin = datetime.date(2021,1,3)
    end = datetime.date(2021,1,3)
    download_data_to_image(begin,end,False,True)
    begin = datetime.date(2021,1,4)
    end = datetime.date(2021,1,4)
    download_data_to_image(begin,end,False,False)
    
