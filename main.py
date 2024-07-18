import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Define categories
Categories = ['Bacterial Red disease', 'Bacterial gill disease', 'Healthy Fish']

# Initialize data lists
flat_data_arr = []  # Input array
target_arr = []  # Output array

# Directory containing the images
datadir = "Freshwater Fish Disease Aquaculture in south asia/Train"
# Load and process images
for category in Categories:
    print(f'Loading... category: {category}')
    path = os.path.join(datadir, category)
    for img_name in os.listdir(path):
        img_array = imread(os.path.join(path, img_name))
        img_resized = resize(img_array, (150, 150, 3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(category))
    print(f'Loaded category: {category} successfully')

# Convert lists to arrays
flat_data = np.array(flat_data_arr)
target = np.array(target_arr)

# Create DataFrame
df = pd.DataFrame(flat_data)
df['Target'] = target

# Split data into training and testing sets
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=77, stratify=y)

# Define parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.0001, 0.001, 0.1, 1],
    'kernel': ['rbf', 'poly']
}
# Create SVM classifier
svc = svm.SVC(probability=True)
# Create model using GridSearchCV
model = GridSearchCV(svc, param_grid)
# Train the model
model.fit(x_train, y_train)

# Save the model to a file
joblib.dump(model, 'fish_disease_model.pkl')

# Evaluate the model
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_pred, y_test)
print(f"The model is {accuracy*100}% accurate")
print(classification_report(y_test, y_pred, target_names=Categories))






# import pandas as pd
# import os
# import time
# from skimage.transform import resize
# from skimage.io import imread
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import svm
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report
# Categories=['Bacterial Red disease','Bacterial gill disease']
# flat_data_arr=[] #input array
# target_arr=[] #output array
# datadir="Freshwater Fish Disease Aquaculture in south asia/Train"
# #path which contains all the categories of images
# for i in Categories:
#     print(f'loading... category : {i}')
#     j=0;
#     path=os.path.join(datadir,i)
#     for img in os.listdir(path):
#         j=j+1
#         if(j==17):
#             break
#         img_array=imread(os.path.join(path,img))
#         img_resized=resize(img_array,(150,150,3))
#         flat_data_arr.append(img_resized.flatten())
#         target_arr.append(Categories.index(i))
#     print(f'loaded category:{i} successfully')
# flat_data=np.array(flat_data_arr)
# target=np.array(target_arr)
# print("lplp")
#
# df=pd.DataFrame(flat_data)
# df['Target']=target
# print(df.shape)
#
#
# #input data
# x=df.iloc[:,:-1]
# #output data
# y=df.iloc[:,-1]
#
# # Splitting the data into training and testing sets
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20, random_state=77,stratify=y)
#
# # Defining the parameters grid for GridSearchCV
# param_grid={'C':[0.1,1,10,100],
# 			'gamma':[0.0001,0.001,0.1,1],
# 			'kernel':['rbf','poly']}
#
# # Creating a support vector classifier
# svc=svm.SVC(probability=True)
#
# # Creating a model using GridSearchCV with the parameters grid
# model=GridSearchCV(svc,param_grid)
#
# # Training the model using the training data
# model.fit(x_train,y_train)
# print("jiji")
#
# y_pred = model.predict(x_test)
#
# # Calculating the accuracy of the model
# accuracy = accuracy_score(y_pred, y_test)
#
# # Print the accuracy of the model
# print(f"The model is {accuracy*100}% accurate")
#
# print(classification_report(y_test, y_pred, target_names=['Bacterial Red disease','Bacterial gill disease']))
# # predict
# path = "Freshwater Fish Disease Aquaculture in south asia/Test/Bacterial Red disease"
# contents = os.listdir(path)
#
# for i in contents:
#     img_path = os.path.join(path, i)
#     img = imread(img_path)
#     plt.imshow(img)
#     #plt.show()  # Display the image
#     img_resize = resize(img, (150, 150, 3))
#     l = [img_resize.flatten()]
#     probability = model.predict_proba(l)
#     for ind, val in enumerate(Categories):
#         print(f'{val} = {probability[0][ind]*100}%')
#     print("The predicted image is : " + Categories[model.predict(l)[0]])
#     time.sleep(1)
