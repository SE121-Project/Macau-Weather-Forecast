# *DecisionTree RandomForest*
This project use Decision tree to simulate and predict the rain condition in some place (mainly simulate the wether in Macao)


## **Content**
- [Background](#Background)
- [Working method](#Working-method)
    - [Data collecting](#Data-collecting)
    - [Data cleaning](#Data-cleaning)
    - [Model training](#Model-traning)
- [Environment](#Environment)
- [Badge](#Badge)
- [Members](#Members)
    - [Developer](#Developer)
    - [Participators](#Participators)


## **Background**
We found plenty of example about weather prediction when we study machine learning. After understanding the relevant information, we made a model of using the classical random forest in machine learning and simulated the rainfall situation in a region.

## **Working-method**
In this project we use decision tree in *sklearn* library (sklearn.tree.DecisionTreeClassifier) to predict the rain in Macao. 

- Data collecting
    - We download the observation data about rainfall of Macao during 2020-12-4 to 2021-12-4 from [SMG - Bureau of Geophysics and Meteorology](https://www.smg.gov.mo).
- Data cleaning
    - Some data of rainfall has been missing that can't be used to train model. So we set ```missing_values = ["--"]``` and use ```.dropna()``` to drop relative data.
- Model training
    - We use air pressuse (*air_pressure*), highest temperature (*high_tem*), average temperature (*aver_tem*), lowest temperature (*low_tem*), relative humidity (*relative_humidity*), sunlight time (*sunlight_time*), wind direction (*wind_direction*), wind speed (*wind_speed*) and rain accumulation (*rain_accumulation*) as all the features to train model.
    - We take 33% of the data as training set and set 10 as the maximum nodes of leaves to prevent overfitting.
    - Then take the last data as the testing set to simulate the rainfall condiation.
- Testing
    - The final test show the accuracy reaches 86% and 88%.
    ![predict result](./result.png)

## **Environment**
- scikit-learn
- pandas
- matplotlib
- seaborn
- joblib


## **Developer**
- Developer
    [@IvanMao714](https://huggingface.co/IvanMao714)

- Participators 
    [@FengD](https://huggingface.co/FengD)  [@ZiYu]()

---
title: DecisionTree RandomForest  
emoji: 📚  
colorFrom: purple  
colorTo: pink  
sdk: gradio  
sdk_version: 3.16.2  
app_file: app.py  
pinned: false  
license: ecl-2.0  
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
