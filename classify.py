"""
This program is responsible for the font classification by using the data in the 'models'
directory produced by 'train.py' to make predictions on new data

The method for classification is the following:
    1. the word is broken up into the characters that form it, and concentrated images
        are formed (using 'concentrateImg' defined in 'dataPrepper.py')
    2. for each character, a model is picked from the 'models' directory: if the
        character has its own model, it is picked; otherwise - the global one is picked.
    3. the output of the selected model is evaluated (for each character on its own model).
    4. the weighted sum of the output vectors is calculated (using the
        weights stored in 'models/weights.h5'), and the highest component is selected
        as the classification output.
    5. produces an output file results_csv_file_name containing information about the predicted font of every character
       appearing in the input dataset.
"""

from utils import font_index_map, num_of_fonts, PROJECT_PATH
from dataPrepper import processDataset
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session as keras_clear_session
from tensorflow.compat.v1 import logging
import numpy as np
import h5py
import csv
import gc


# names for the different fields in the output csv file.
fieldnames = ['', 'image', 'char', 'Alex Brush', 'Open Sans',
              'Sansation', 'Ubuntu Mono', 'Titillium Web']

# results csv file name
results_csv_file_name = 'results.csv'

# the size of the partitions of the test set we are going to process individually
batch_size = 100


def rawEvaluate(images, by_char_indexing, weighted=False, verbose=True):
    """
    rawEvaluate : evaluates the raw output of the models in the 'models' directory.

    Parameters:
        > images (numpy.ndarray): a numpy array of shape (N, 64, 64, 3), representing a list of N 64-by-64
            pixels image of characters.
        > by_char_indexing (dict): a dictionary indexed by strings containing ASCII values of characters.
            Each entry should be a 1-dimensional numpy array of indices of images of the corresponding
            character.
        > weighted (bool, optional): a boolean, representing weather or not output vectors should be weighted.
        > verbose (bool, optional): a boolean, representing weather or not you wish to see the process documented
            in real time in the terminal.

    Returns:
        numpy.ndarray: a N-by-'num_of_fonts' numpy array. The [i, j] element of it is the predicted probability
            of the font of the character in the i'th image being the j'th font.
            If 'weighted' is True, each output vector is multiplied by the weight of the model that generated it.

    Requirements:
        The 'models' directory (produced by 'train.py'), must be in the PROJECT_PATH directory.
    """

    # get fundamental information:
    N = images.shape[0]
    weights_file = h5py.File(PROJECT_PATH + 'models/weights.h5', 'r')

    results = np.empty((N, num_of_fonts)) # numpy array to hold the results
    processed = [] # list of indices already-processed examples.

    # prevent warning on repeatedly calling predict 
    logging.set_verbosity(logging.ERROR)

    # process images of characters having an individual model in 'models' directory 
    for file_name in weights_file.keys():
        if file_name != 'global' and file_name != 'weights.h5' and file_name in by_char_indexing.keys():
            model_name = file_name

            # evaluate output vectors on those images:
            if verbose: print('\t> Process examples of character \'' + chr(int(model_name)) + '\':', end=' ')
            if verbose: print('Load model...', end=' ')
            indices = by_char_indexing[model_name][()]
            model = load_model(PROJECT_PATH + 'models/'+model_name)
            if verbose: print('done. Evaluate raw output...', end=' ')
            results[indices] = model.predict(images[indices])
            if weighted: results[indices] *= weights_file[model_name]
            if verbose: print('done.')

            # record those images as processed
            processed += indices.tolist()

            # cleanup
            keras_clear_session()
            del model
            gc.collect()

    # now, use the global model to evaluate output for the rest of the examples:
    if verbose: print('\t> Rest of examples: Load model...', end=' ')
    indices = [i for i in range(N) if i not in processed]
    model = load_model(PROJECT_PATH + 'models/global')
    if verbose: print('done. Evaluate raw output...', end=' ')
    results[indices] = model.predict(images[indices])
    if weighted: results[indices] *= weights_file['global']
    if verbose: print('done.')

    # cleanup
    keras_clear_session()
    del model
    gc.collect()

    if weighted: weights_file.close()
    return results


def evaluateWightedVectors(images, by_char_indexing, by_word_indexing, verbose=True):
    """
    evaluateWightedVectors : evaluates predicted probability vectors.
        Parameters:
            > 'images' - a numpy array of shape (N,64,64,3), representing a list of N 64-by-64
                pixels image of characters (concentrated using the 'concentrateImg' function defined
                in 'dataPrepper.py').
            > 'by_char_indexing' a dictionary (or dictionary-like) indexed by strings containing
                ASCII values of characters. Each entry should be a 1-dimensional numpy array (or
                array-like) of indices of images of the corresponding character.
            > 'by_word_indexing' a list, or a dictionary (or a dictionary-like) indexed arbitrarily.
                Each entry should be a 1-dimensional numpy array (or array-like) of indices of examples
                of the same word. Should cover all the examples.
            > 'verbose' - a boolean, representing wether or not you wish to see the process documented
                in real time in the terminal.
        Returns:
            An N-by-'num_of_fonts' numpy array. The [i,j] element of it is the predicted probability of the
            font of the character in the i'th image being the j'th font (in the 'font_index_map'
            array defined in 'dataPrepper.py').
        Requirements:
            The 'models' directory (produced by 'train.py'), being in PROJECT_PATH directory.
    """

    # numpy array to hold the results
    results = np.empty((images.shape[0], num_of_fonts))

    # evaluate weighted output vectors
    if verbose: print('\tEvaluating raw output...')
    evaluations = rawEvaluate(images, by_char_indexing, weighted=True, verbose=verbose)
    if verbose: print('\tDone evaluating raw output.')
    if verbose: print('\tEvaluating final output...', end=' ')

    # make sure 'by_word_indexing' is a list (and not dictionary/dictionary-like indexed arbitrarily)
    if not isinstance(by_word_indexing, list):
        by_word_indexing = by_word_indexing.values()

    # process each word
    for indices in by_word_indexing:
        indices = indices[()]  # convert to numpy array (if it is array-like)

        # calculate normalized weighted sum of estimated probability vectors
        weighted_sum = np.sum(evaluations[indices], axis=0)
        norm_weighted_sum = weighted_sum / max(np.sum(weighted_sum),1e-100)
        results[indices] = norm_weighted_sum

    if verbose: print('done.')
    return results


def predict(images, by_char_indexing, by_word_indexing, verbose=True):
    """
    predict : classifies examples to fonts.
        Parameters:
            > 'images' - a numpy array of shape (N,64,64,3), representing a list of N 64-by-64
                pixels image of characters (concentrated using the 'concentrateImg' function defined
                in 'dataPrepper.py').
            > 'by_char_indexing' a dictionary (or dictionary-like) indexed by strings containing
                ASCII values of characters. Each entry should be a 1-dimensional numpy array (or
                array-like) of indices of images of the corresponding character.
            > 'by_word_indexing' a list, or a dictionary (or a dictionary-like) indexed arbitrarily.
                Each entry should be a 1-dimensional numpy array (or array-like) of indices of examples
                of the same word. Should cover all the examples.
            > 'verbose' - a boolean, representing wether or not you wish to see the process documented
               in real time in the terminal.
        Returns:
            A numpy 1-dimensional array of N font_indexs. The i'th font_index is the predicted font_index for
            the character appearing in the i'th image (see 'font_index_map' in dataPrepper.py).
         Requirements:
                The 'models' directory (produced by 'train.py'), being in PROJECT_PATH directory.
    """

    # calculate probability vectors
    probabilities = evaluateWightedVectors(images, by_char_indexing, by_word_indexing, verbose=verbose)

    # return font_index of the highest probability
    return np.argmax(probabilities, axis=1)


def dumpResultsToCSV(predictions, chars, by_img_indexing, writer, in_writer_index=0):
    """
    dumpResultsToCSV : dumps font classifications (and other info) to csv file.
        Parameters:
            > 'predictions' - list of N font_indexs. The i'th font_index is the predicted font_index for i'th
                example in testing set.
            > 'chars' - list of N ASCII values. The i'th value is the ASCII value of i'th example
                in testing set.
            > 'by_img_indexing' - a dictionary indexed by image names. Each entry in it is the
                index of the first example in test set that belongs to the corresponding image
                (the rest are the ones that follow).
            > 'writer' - csv writer to write the results to.
            > 'in_writer_index' - the index of the next row in the output writer.

        Produces an output csv file named by results_csv_file_name that contains information about
        the predictions.
    """

    # organize the image names in ascending order of appearances in test set
    img_names = list(by_img_indexing.keys())
    img_names.sort(key=lambda name: by_img_indexing[name])

    # setup accounting for checking the current image name
    N = len(predictions)
    curr_img_name = None if not img_names else img_names.pop(0)
    next_img_index = N if not img_names else by_img_indexing[img_names[0]]

    # process each example
    for i, (font_index, char) in enumerate(zip(predictions, chars)):
        # check if image name should be changed
        if i == next_img_index:
            curr_img_name = None if not img_names else img_names.pop(0)
            next_img_index = N if not img_names else by_img_indexing[img_names[0]]

        # record information regarding the current example in output file
        writer.writerow({
                '':               in_writer_index+i,
                'image':           curr_img_name,
                'char':            chr(char),
                # use predicted font_index and the font_index map to determine font
                'Alex Brush':        1 if font_index_map[font_index] == b'Alex Brush' else 0,
                'Open Sans':      1 if font_index_map[font_index] == b'Open Sans' else 0,
                'Sansation':         1 if font_index_map[font_index] == b'Sansation' else 0,
                'Ubuntu Mono':    1 if font_index_map[font_index] == b'Ubuntu Mono' else 0,
                'Titillium Web':       1 if font_index_map[font_index] == b'Titillium Web' else 0
                })


def partitionList(src_list, partition_size):
    """
    devideToBatches : returns a list of partitions of a given list, with the given partition size.
    """

    partitions = []  # to hold the results
    start = 0  # index into the beginning of the current partition
    while start < len(src_list):
        end = min(start + partition_size, len(src_list))
        partitions.append(src_list[start : end])
        start = end
    return partitions


def classify_main(test_file_name):
    """
    Classify_main : uses the input dataset found at test_file.h5, and produces an output results_csv_file_name
    containing information about the predicted font of every character appearing in it.

    Requirements:
    The 'models' directory produced by 'train.py',  and the test_file_name being in the PROJECT_PATH directory.
    """

    # set output dictionary writer
    with open(PROJECT_PATH + results_csv_file_name, 'w', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames)
        writer.writeheader()
        in_writer_index = 0

        # load the test set and partition it to batches
        with h5py.File(PROJECT_PATH + test_file_name, 'r') as db:
            img_names = list(db['data'].keys())
            img_names_batches = partitionList(img_names, batch_size)

            for batch_i, img_names_batch in enumerate(img_names_batches):
                print('Process batch', batch_i+1, '/', len(img_names_batches), '...')

                # preprocess (concentrate the images on the characters in them)
                print('Preprocessing data...')
                processed_data = processDataset(db, img_names_batch, fonts_included=False)
                (imgs, chars, _, by_char_indexing, by_word_indexing, by_img_indexing) = processed_data
                print('Done preprocessing data for bacth', batch_i+1, '/', len(img_names_batches),'.\n')

                # make predictions
                print('Making predictions...')
                predictions = predict(np.array(imgs), by_char_indexing, by_word_indexing)
                print('Done making predictions for bacth', batch_i+1, '/', len(img_names_batches),'.\n')

                # dump results to csv file
                print('Dumping results to' + results_csv_file_name + '...', end=' ')
                dumpResultsToCSV(predictions, chars, by_img_indexing, writer, in_writer_index)
                in_writer_index += len(imgs)
                print('done.', end=' ')

                # cleanup
                print('Cleaning up...', end=' ')
                del (imgs, chars, _, by_char_indexing, by_word_indexing, by_img_indexing)
                del processed_data
                del predictions
                gc.collect()
                print('done.')

                print('Done processing batch', batch_i+1, '/', len(img_names_batches), '.\n')

    print('')
    print('FINISH')
