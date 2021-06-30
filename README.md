# Bachelor_Proj

The repository is organised as follows:
- dataloader, utils, models, checkpoint and resutlts are mainly files from https://github.com/mileyan/AnyNet, some of them modified to fit our dataset.
- app.py and apply.py are files to output the image after applying the model finetuned on them (the first one using the default loader and the second one using my modified loader).
- Dpeth_Estimation.py is the file I used to run the finetuning and to apply the model on the image.
- Preprocessing.py is the file I used to develop the code to match the stereo image input with the Lidar.
