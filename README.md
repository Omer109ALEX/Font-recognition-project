# Font Recognition Project
The final project at computer vision, my grade is `96`

A program for recognizing fonts in text appearing in photos - the photos where created by https://github.com/ankush-me/SynthText
![image](https://github.com/user-attachments/assets/07de67a0-337f-44b1-9048-472af7180240)


To run the project follow the next steps:
1. Save the project files in the same directory.
2. Change PROJECT_PATH in `utils.py` according to the project directory.
3. Change the train and test files names (train_file_name, test_file_name) in `utils.py`.
4. run `main.py`.
    + to train from zero the model use run_project(need_train=True).
    + to visualize the results use run_project(need_visualize=True).


The project files:
+ `utils.py` - consts and parameters for other programs.
+ `main.py` - for run the project with the correct parameters.
+ `dataPrepper.py` - for preprocessing the data and splitting it to train, test, and validation partitions. The output of it for our data `"ReadySynthText.h5"` can be found [here](https://drive.google.com/file/d/1-2oiOvT17IcqxVPW1WUr67zhywSRdMRF/view?usp=share_link).
+ `train.py` - for training the model(s). The output `models` directory can be found [here](https://drive.google.com/drive/folders/16-LBT4u3U803QTWa25hfKgOUZ4-fjxOw?usp=sharing).
+ `classify.py` - to produce predictions for an unlabeled test set. the output on the testing data below `"results.csv"` can be found [here](https://drive.google.com/file/d/1qTR1PgJnWgjXQOxEi0Q_8sr7t39_Lh8t/view?usp=sharing)
+ `visualizeResults.py` - to visualize the system's performance on a labeled set, using various metrics and plots.


The training data for this project can be found [here](https://drive.google.com/file/d/1zZ2wiOGacEMtgM9VsFsP9g2Iug9CfVzM/view?usp=share_link).

The testing data for this project can be found [here](https://drive.google.com/file/d/1YwLcXqLArFSOtoepQw7nC1t4jC8CFxpI/view?usp=sharing).


All project files can be found [here](https://drive.google.com/drive/folders/179PUMEVEeKKPSA2gZ-nhxMNtaDlpJ4vo?usp=sharing).


