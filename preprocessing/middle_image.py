import csv
import os
from pathlib import Path

import scipy.misc
from loguru import logger

import imageio
from preprocessing.base_preprocessing import BasePreprocessing


class MiddleImage(BasePreprocessing):
    def __init__(self):
        BasePreprocessing.__init__(self)

    def run(self):
        logger.info("Running preprocessing")
        root = Path(self.input_dir)
        labels = []

        for sign_dir in root.iterdir():
            if sign_dir.is_dir():
                name = sign_dir.stem
                logger.info("Processing {}".format(name))
                labels += self._collect_signs_img(sign_dir, name)

        logger.info("Writting labels CSV")
        with open("label.csv", "w") as out:
            csv_out = csv.writer(out)
            csv_out.writerow(["path", "label"])

            for row in labels:
                csv_out.writerow(row)

    def _collect_signs_img(self, signs_path, label):
        dest = "{}/{}".format(self.output_dir, label)

        if not os.path.exists(dest):
            os.makedirs(dest)

        folder = Path(signs_path)
        all_treated = []

        for elem in folder.iterdir():
            if elem.is_file():
                try:
                    all_treated.append(self._process_sign(elem, dest, label))
                except ValueError as e:
                    continue

        return all_treated

    def _process_sign(self, sign, dest, label):

        sign_name = sign.name.split(".")[0]
        img_path = "{}/{}.png".format(dest, sign_name)

        if os.path.exists(img_path):
            return (img_path, label)

        gif = imageio.mimread(str(sign), memtest=False)
        selected = int(len(gif) / 2)

        img = gif[selected]
        scipy.misc.imsave(img_path, img)

        return (img_path, label)
