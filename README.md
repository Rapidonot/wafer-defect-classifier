
Semiconductor Wafer Defect Pattern Classifier

A deep learning framework that can identify the defects in wafer bin maps (WBM) and classify them

The model recognizes 8 semiconductor wafer defect signature patterns:
- Centre
- Donut
- Edge-Loc
- Edge-Ring
- Loc
- Near-full
- Random
- Scratch


The model was trained using ResNet18 with Binary Cross Entropy and supports multi label classfication using a sigmoid function

A combined dataset of WM-811k and MixedWM38 was used to train the model , with 70% used for training and 15% for validation 

It achieved a macro F1 score of 0.97 from testing on the last 15% of the combined dataset


Linked here is a working demo website 

PNGs,JPEG,WEPG files are supported , but please make sure images are in Viridis colour palette and are WBMs 

https://rapidonot-wafer-defect-classifier-dashboardapp-zleqcs.streamlit.app


To run the training script , enter this into your code terminal ( use python 3 for mac )

python modeltraining.py --model resnet18 -- loss bce 

I have also include training scripts for VanillaCNN inside the code with option for focal loss for either model

You can run them using

python modeltraining.py --model vanilla_cnn --loss bce ( use python 3 for mac ) 


