import os

import pydicom
from PIL import Image
import numpy as np

from bs4 import BeautifulSoup
import csv

import sys


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

            print(uid_and_imageid, manu)
    return uid_and_imageid_to_manu



def get_uid_imgid_to_suvr(data_path, label_file):

    uid_to_suvr = {}
    with open(label_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            uid = row['RID']
            uid = uid.zfill(4)  # fill 0's to the front

            if uid in uid_to_suvr:
                uid_to_suvr[uid].append(float(row['SUMMARYSUVR_WHOLECEREBNORM']))
            else:
                uid_to_suvr[uid] = [float(row['SUMMARYSUVR_WHOLECEREBNORM'])]

    base_dir = data_path
    f_out = open('labels.txt', "w")
    for adni_chunk in os.listdir(base_dir):
        path = os.path.join(base_dir, adni_chunk)
        subject_ids = os.listdir(path)
        for subject_id in subject_ids:
            uid = (subject_id.split('_'))[2]

            path2 = os.path.join(path, subject_id)
            f = os.path.join(path2, os.listdir(path2)[0])
            dates = sorted(os.listdir(f))  # sort the date is very important

            for visit, date in enumerate(dates):

                image_id = os.listdir(os.path.join(f, date))[0]
                if uid in uid_to_suvr:
                    suvr = uid_to_suvr[uid][visit]

                    print(uid + "," + image_id + "," + str(suvr) + "\n")
                    f_out.write(uid + "," + image_id + "," + str(suvr) + "\n")
    f_out.close()

get_uid_imgid_to_suvr(sys.argv[1], sys.argv[2])
