# Convolutional Neural Networks
This dataset is too large to host on GitHub --> ~10,000 images. Instead, the structure is detailed below:
```
dataset
\ single_prediction  # Any images
\ \ cat_or_dog_1.jpg
\ \ cat_or_dog_2.jpg
\ \ cat_or_dog_3.jpg
\ \ cat_or_dog_4.jpg

\ test_set
\ \ cats # 1,000 images
\ \ \ cat.4001.jpg
\ \ \ cat.4002.jpg
\ \ \ ...
\ \ dogs # 1,000 images
\ \ \ dog.4001.jpg
\ \ \ dog.4002.jpg
\ \ \ ...

\ training_set
\ \ cats # 4,000 images
\ \ \ cat.1.jpg
\ \ \ cat.2.jpg
\ \ \ ...
\ \ dogs # 4,000 images
\ \ \ dog.1.jpg
\ \ \ dog.2.jpg
\ \ \ ...
```
Any dataset should suffice. Example:
[Kaggle's cats-vs-dogs dataset (https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset/data)](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset/data)