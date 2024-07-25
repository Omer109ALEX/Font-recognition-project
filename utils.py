"""
------------------- CHANGE THOSE PARAMETERS FOR YOUR NEEDS----------------------
"""
PROJECT_PATH = '/content/drive/MyDrive/Project/'  # names of the project path
train_file_name = 'SynthText_train.h5'  # names of train file
test_file_name = 'SynthText_train.h5'  # names of test file
"""
---------------------------------------------------------------------------------
"""

# list of fonts, indexed by their font_index (font index).
font_index_map = [b'Alex Brush', b'Open Sans', b'Sansation', b'Ubuntu Mono', b'Titillium Web']
num_of_fonts = font_index_map.__len__()

# number of epochs for global model
NUM_OF_GLOBAL_EPOCHS = 150
# number of epochs for individual models
NUM_OF_INDIVIDUAL_EPOCHS = 50
# number of minimum images per character to make an individual model
MIN_FOR_INDIVIDUAL_MODEL = 20