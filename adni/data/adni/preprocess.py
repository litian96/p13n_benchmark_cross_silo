import os

import pydicom
from PIL import Image
import numpy as np
import sys
import csv

from bs4 import BeautifulSoup

manufacturer_name_mapping={'CPS':'CPS',
                           'GE MEDICAL SYSTEMS': 'GE_MEDICAL_SYSTEMS',
                           'GEMS': 'GEMS',
                           'HERMES': 'HERMES',
                            'MiE': 'MiE',
                           'Multiple': 'Multiple',
                           'Philips': 'Philips',
                            'Philips Medical Systems': 'Philips_Medical_Systems',
                           'SIEMENS': 'SIEMENS',
                            'Siemens ECAT': 'Siemens_ECAT',
                           'Siemens/CTI': 'Siemens_CTI'}


def get_names(path):
    names = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext in ['.dcm']:
                names.append(filename)

    return names


def convert_dcm_jpg(name):
    im = pydicom.dcmread(name)

    im = im.pixel_array.astype(float)

    rescaled_image = (np.maximum(im, 0) / im.max()) * 255  # float pixels
    final_image = np.uint8(rescaled_image)  # integers pixels
    final_image = Image.fromarray(final_image)

    return final_image



def get_uid_manufacturer(meta_data_path):

    files = os.listdir(meta_data_path)

    uid_and_imageid_to_manu = {}

    for f in files:
        if f.endswith('xml'):
            infile = open(os.path.join(meta_data_path, f), "r")
            tree = BeautifulSoup(infile.read(), features="xml")
            manu = tree.project.subject.study.series.seriesLevelMeta.relatedImageDetail.originalRelatedImage.protocolTerm.protocol.get_text()

            f = f[:-4].split("_")
            uid = f[3]
            imageid = f[-1]
            uid_and_imageid = uid+"_"+imageid
            uid_and_imageid_to_manu[uid_and_imageid] = manu

    return uid_and_imageid_to_manu


meta_data_path, raw_data_path, png_path = sys.argv[1], sys.argv[2], sys.argv[3]

# get mapping from userid_imageid to manufacturer
base_dir = meta_data_path  # meta_data_path
uid_and_imageid_to_manu = {}
dict = get_uid_manufacturer(base_dir)
uid_and_imageid_to_manu = {**uid_and_imageid_to_manu, **dict}
#
print('finish extracting manufacturers...')



base_dir = raw_data_path
png_files = png_path

for vals in manufacturer_name_mapping.values():
    os.mkdir(os.path.join(png_files, vals))
for adni_chunk in os.listdir(base_dir):
    print(adni_chunk)
    path = os.path.join(base_dir, adni_chunk)
    subject_ids = os.listdir(path)
    for subject_id in subject_ids:
        #print(subject_id)
        uid = (subject_id.split('_'))[2]
        path2 = os.path.join(path, subject_id)
        f = os.path.join(path2, os.listdir(path2)[0])
        dates = os.listdir(f)
        print(uid)
        for date in dates:
            image_id = os.listdir(os.path.join(f, date))[0]
            path3 = os.path.join(f, date, image_id)
            images = os.listdir(path3)  # multiple slices
            uid_and_imageid = uid + "_" + image_id
            manufacturer = uid_and_imageid_to_manu[uid_and_imageid]
            manufacturer = manufacturer_name_mapping[manufacturer]
            for image in images:
                slice_number = int(image.split("_")[-3])
                if slice_number >= 50 and slice_number < 57:
                    image_png = convert_dcm_jpg(os.path.join(path3, image))
                    image_png.save(png_files + manufacturer + "/" + image[:-4] + '.png')

