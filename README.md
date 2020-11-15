# Requirements
python3.6
```
pip install https://download.pytorch.org/whl/cu90/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
pip install torchvision
pip install tqdm
pip install tensorflow
pip install tensorboardX
pip install easydict
```

# How to run?
```
# FOR IMAGENET:
change the "train_dir" in /configs/ImageNet/resnet50/SNIP50.json to the folder with imagenet downloaded, i.e. where the train/val folder is
change the "PATH_TO_DATA" in IMNET_Stable09 as well as "PATH_TO_CHECKP" (see bash script)
Then run:

 bash IMNET_Stable09.sh
 bash IMNET_Stable095.sh
 bash IMNET_SNIP09.sh
 bash IMNET_SNIP095.sh
 ```

