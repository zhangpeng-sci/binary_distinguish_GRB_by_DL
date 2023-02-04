# binary_distinguish_GRB_by_DL

## requirements

numpy==1.15.4  
kears==2.14   
tensorflow==1.12.0  
keras_contrib

```
how to install keras_contrib:  
--------------- 
git clone https://www.github.com/keras-team/keras-contrib.git  
cd keras-contrib  
python setup.py install  
---------------
or use keras source code (keras_contrib directory) directly

```

## datasets(numpy: npy file)

download link:

## trained model(h5 file)

download link:  
best model (ResNet-CBAM@64ms): [h5 file](./trained_model/resnet-CBAM_64ms.h5)

## candidates

candidate list: [csv](./candidates/candidate_list_20221221.csv)  
image of candidate list: [rar](./candidates/img_of_candidate_list_20221221.rar)  
example:  
![210702A](./ref_file/candidate_210702A_2021-07-02T002344.png)
left:  mapping-curves of feature    
right: heat map of features

## how to train model

see jupyter-notebook: [ipynb](./code/train_model.ipynb)

## how to test the already trained model

see jupyter-notebook: [ipynb](./code/test_model.ipynb)



