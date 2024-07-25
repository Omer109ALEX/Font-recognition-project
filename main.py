from dataPrepper import dataPrepper_main
from train import train_main
from classify import classify_main
from visualizeResults import visualizeResults_main
from utils import test_file_name


def run_project(need_train=False, need_visualize=False):
    """
    Run_project : run the font recognition project

        NOTE: before you run the function, check that the parameters on top are correct

        Parameters:
            > 'need_train' - (boolean) if needed to train from zero a new model.
            > 'need_visualize' - (boolean) if needed to visualize the model results.
    """

    # first prepper the data and then train the models
    if need_train:
        dataPrepper_main()
        train_main()

    # classify the test DataSet and create results.csv file
    classify_main(test_file_name)

    # visualize the results
    if need_visualize:
        visualizeResults_main()


if __name__ == '__main__':
    run_project()

