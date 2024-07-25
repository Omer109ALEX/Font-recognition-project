"""
This program is responsible for processing the dataset found at 'SynthText.h5', and
building up a "ready" dataset, organized in a specific hierarchy for later steps of
the process.

The dataPrepper process includes the following:
* Randomly split the images to 20% test, 20% validation, and 60% training.
* Concentrate the images on the different characters presented in them.
* Organize the data into an HDF5 file 'ReadySynthText.h5', as described below.

The resulting "ready" data file has the following form:
* there are 3 main groups: 60% 'train', 20% 'test', 20% 'val'.
* Each of those has 2 subgroups: 'data' and 'indexing'.

* 'data' includes the data itself:
    > 'images' (N,64,64,3) - concentrated images.
    > 'font_index' (N) - labels of the images (represented by font_index).
    > 'chars' (N) - containing the ASCII values of the characters in the images.

* 'indexing' is used for convenient access into the datasets in 'data':
    > 'by_char' - all images of the same character named after the ASCII value.
    > 'by_word' - all images of the same word that are named arbitrarily by a serial number.
    > 'by_img' - all images derived from the same original image

"""

from utils import PROJECT_PATH, train_file_name, font_index_map
import numpy as np
import cv2
import h5py
from math import ceil
import random


def concentrateImg(src_img, src_BB, target_BB_res=(44, 44), margins=10):
    """
    concentrateImg : "concentrates" an image on a bounding box.
        map the source image by using homography
        between the boundary box and the target resolution box and add margins.

        Parameters:
            > 'src_img' - source image.
            > 'src_BB' - 2x4xm matrix whose columns are the vertices of a bounding box.
            > 'target_BB_res' - resolution to map the boundary box to.
            > 'margins' - number of pixels to leave as margins
        Returns:
            A numpy array holding an image of resolution:
            (2*margins + target_BB_res[0], 2*margins + target_BB_res[1]),
    """

    # target boundary box (for the homography)
    target_BB = np.array([[margins, margins],
                          [margins + target_BB_res[0], margins],
                          [margins + target_BB_res[0], margins + target_BB_res[1]],
                          [margins, margins + target_BB_res[1]]]).transpose()

    # resolution of the target concentrated image (including margins)
    target_img_res = (2 * margins + target_BB_res[0], 2 * margins + target_BB_res[1])

    # calculate the homography from original BB to target BB
    H = cv2.findHomography(src_BB.transpose(), target_BB.transpose())[0]
    return cv2.warpPerspective(src_img, H, target_img_res)


def splitList(ls, partition1, partition2):
    """
    splitList : splits a list into 3 partitions
    """

    # calculate the ending point of the first 2 parts
    end1 = ceil(partition1 * len(ls))
    end2 = ceil((partition1 + partition2) * len(ls))
    # split
    part1 = ls[:end1]
    part2 = ls[end1:end2]
    part3 = ls[end2:]
    return part1, part2, part3


def processDataset(db, img_names, fonts_included=True, verbose=True):
    """
    Preprocesses synthetic text image data stored in an HDF5 file.

    Parameters:
        > db (HDF5 file): The HDF5 file holding the data.
        > img_names (list of str): List of names of images in the dataset that need to be processed.
        > fonts_included (bool, optional): Indicates if the font label information is included in the dataset.
            Defaults to True.
        > verbose (bool, optional): Indicates if the process will be documented in real-time at the terminal.
            Defaults to True.

    Returns:
        tuple: A tuple containing the following elements:
            > result_images (list of numpy arrays): A list of concentrated (numpy) images of individual characters.
            > result_chars (list of str): A list of the characters appearing in each concentrated image.
            > result_font_index (list of ints): A list of the font_index of the characters in each concentrated image
                (this may be None if 'fonts_included' is False).
            > by_char_indexing (dict): A dictionary indexed by strings representing ASCII values of
                the characters appearing the processed images. Each entry in it is a numpy list
                of indices of concentrated images of the corresponding character.
            > by_word_indexing (list of numpy arrays): A list of numpy lists of indices of concentrated
             images of the same word.
            > by_img_indexing (dict): A dictionary indexed by image names. Each entry in it is the
                index of the first concentrated image derived from the corresponding (original) image in
                numpy 0-dimensional array (the rest of the concentrated images derived from this image
                are the concentrated images that follow).
    """

    # initialize outputs
    result_images = []
    result_chars = []
    result_font_index = [] if fonts_included else None
    by_char_indexing = {}
    by_word_indexing = []
    by_img_indexing = {}

    global_curr_i = 0  # index of current result image

    # process each image:
    for img_i, img_name in enumerate(img_names):
        if verbose: print('\t> Preprocessing image', img_i + 1, '/', len(img_names), '...', end=' ')
        local_curr_i = 0  # index of current character processed in the current image

        # extract the image and data about the characters
        img = db['data'][img_name][:] / 255.0
        words = db['data'][img_name].attrs['txt']
        char_BBs = db['data'][img_name].attrs['charBB']
        if fonts_included:
            fonts = db['data'][img_name].attrs['font']

        # process each word:
        for word in words:
            curr_word_indices = np.empty(0, dtype=int)  # indices of characters of this word

            # process each character:
            for char in word:
                # get data about the character
                char_BB = char_BBs[:, :, local_curr_i]
                if fonts_included:
                    font = fonts[local_curr_i]
                    font_index = font_index_map.index(font)

                # concentrate image on the character 
                concentrated_img = concentrateImg(img, char_BB)

                # create entry in the 'by_char' dictionary (if necessary)
                if str(char) not in by_char_indexing.keys():
                    by_char_indexing[str(char)] = np.empty(0, dtype=int)

                # create entry in the 'by_img' dictionary (if necessary)
                if img_name not in by_img_indexing.keys():
                    by_img_indexing[img_name] = np.array(global_curr_i)

                # add the new character image to the results:
                result_images.append(concentrated_img)
                result_chars.append(char)
                if fonts_included:
                    result_font_index.append(font_index)
                # add current example to 'by_char' dictionary
                by_char_indexing[str(char)] = np.append(by_char_indexing[str(char)], global_curr_i)
                # add current example to the current word's indices
                curr_word_indices = np.append(curr_word_indices, global_curr_i)

                global_curr_i += 1
                local_curr_i += 1

            by_word_indexing.append(curr_word_indices)  # add current word to 'by_word' indexing

        if verbose:
            print('done.')

    return result_images, result_chars, result_font_index, by_char_indexing, by_word_indexing, by_img_indexing


def buildDataset(db, img_names, grp):
    """
    BuildDataset : builds a processed version of a subset of a dataset into a HDF5 group.
        Parameters:
            > 'db' - HDF5 (opened) file holding the source dataset.
            > 'img_names' - (list) of names of images in the dataset that need to process.
            > 'grp' - h5py group, into which you wish to load the processed dataset.
        Results:
            The h5py group 'grp' containing processed information about the data in 'db',
            in the format discussed at the top of this file.
    """

    # process the input dataset
    (result_images, result_chars, result_font_index, by_char_results, by_word_results,
     by_img_results) = processDataset(db, img_names)

    # organize the data in the given HDF5 group
    data_sgrp = grp.create_group('data')
    inedxing_sgrp = grp.create_group('indexing')
    by_char_sgrp = inedxing_sgrp.create_group('by_char')
    by_word_sgrp = inedxing_sgrp.create_group('by_word')
    by_img_sgrp = inedxing_sgrp.create_group('by_img')

    data_sgrp.create_dataset('images', data=result_images)
    data_sgrp.create_dataset('font_index', data=result_font_index)
    data_sgrp.create_dataset('chars', data=result_chars)

    for char_ascii in by_char_results.keys():
        by_char_sgrp.create_dataset(char_ascii, data=by_char_results[char_ascii])
    for word_i, indices in enumerate(by_word_results):
        by_word_sgrp.create_dataset(str(word_i), data=indices)
    for img_name in by_img_results.keys():
        by_img_sgrp.create_dataset(img_name, data=by_img_results[img_name])


def dataPrepper_main():
    """
    DataPrepper_main: build-up a processed version of the dataset at 'SynthText.h5'
    in file 'ReadySynthText.h5', in the format described at the top.
    """

    with h5py.File(PROJECT_PATH + train_file_name, 'r') as db:  # import original HDF5 file
        with h5py.File(PROJECT_PATH + 'ReadySynthText.h5', 'w') as new_db:  # create new HDF5 file

            # collect image names and randomly split to train-test-validation (60%-20%-20%)
            names = list(db['data'].keys())
            random.shuffle(names)
            train_names, test_names, val_names = splitList(names, 0.6, 0.2)

            # create main groups
            train_grp = new_db.create_group('train')
            test_grp = new_db.create_group('test')
            val_grp = new_db.create_group('val')

            # fill the groups
            print('Building training set...')
            buildDataset(db, train_names, train_grp)
            print('Done building training set.\n')
            print('Building test set...')
            buildDataset(db, test_names, test_grp)
            print('Done building test set.\n')
            print('Building validation set...')
            buildDataset(db, val_names, val_grp)
            print('Done building validation set.\n')
