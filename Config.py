import os


DATA_DIR = "/home/jasper/Documents/BP_Jasp/data/pg"
IMAGE_DIR = os.path.join(DATA_DIR, "123-130/")
#DATA_DIR = '/home/jasper/Documents/BP_Jasp/data/pg/valSet/sedimentation/'
#IMAGE_DIR = os.path.join(DATA_DIR, "T0132_S002_U010/")
model_dir = '/home/jasper/Documents/project/SiameseChemNet/models/conv_SiamNet'
GrayScale = True
LOG_DIR = '/home/jasper/Documents/BP_Jasp/logs/Siam/'
train_batch_size = 32
train_number_epochs = 50