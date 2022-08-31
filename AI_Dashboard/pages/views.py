from os import POSIX_FADV_RANDOM
from django.http import HttpResponse
from django.shortcuts import render

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import pandas as pd
import numpy as np


df = pd.read_csv("data.csv")

#print(df.head())
#size = df['Area of Evaluation [Department provides comprehensive guidelines to the students in advance by means of a brochure/handbook    ]'].value_counts(sort=1)
#print(size)

#drop not needed cloums
#cloums to drop
dropCloum = [
    'Timestamp',
    'Faculty',
    'Engineering Program',
    'Law Program',
    'Business Program',
    'Arts Program',
    'Other Program',
    'Bachelor  Academic Year in EU',
    'Masters Academic Year in EU',
    'H.S.C or Equivalent study medium',
    'S.S.C (GPA)',
    'H.S.C (GPA)',
    'Did you ever attend a Coaching center?',
    'Coaching center name',
    'Benifits you received from the coaching center',
    '1st Year Semester 1',
    '1st Year Semester 2',
    '1st Year Semester 3',
    '2nd Year Semester 1',
    '2nd Year Semester 2',
    '2nd Year Semester 3',
    '3rd Year Semester 1',
    '3rd Year Semester 2',
    '3rd Year Semester 3',
    '4th Year Semester 1',
    '4th Year Semester 2',
    '4th Year Semester 3',
    'Regular/Irregular',
    'Classes are mostly',
    'Q7. In your opinion,the best aspect of the program is',
    'Q8. In your opinion,the next best aspect of the program is',
    'What aspects of the program could be improved?',
    'Do you feel that the quality of education improved at EU over the last year?',
    'Do you feel that the image of the University improved over the last year?',
    'Username'
]

for ind in dropCloum:
    df.drop([ind], axis=1, inplace = True) #drop the cloum and update df

#handling missing value
df = df.dropna()

#converting non-numeric data to numeric
df.Gender[df.Gender == 'Male'] = 1
df.Gender[df.Gender == 'Female'] = 2

#defining the dependent variable
Y = df['Gender'].values
Y = Y.astype('int')

#defining the independent variable
X = df.drop(labels = ['Gender'], axis = 1)

#spliting datset into train and test dataset
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 40)

#performing randomforest analysis
model = RandomForestClassifier(n_estimators = 10, random_state = 40)
model.fit(X_train, Y_train)

#predicating
predication = model.predict(X_test)
#size = predication.value_counts(sort=1)
#print(predication)
male = 0
female = 0
#counting number of male and female
for gn in predication:
    if gn == 1:
        male = male + 1
    else:
        female = female + 1

#print('male = ', male)
#print('female = ', female)

#testing accuracy
accuracy = metrics.accuracy_score(Y_test, predication)
#print('accuracy = ', accuracy)

#rating of student guidline
guidlin_rating = df['Area of Evaluation [Department provides comprehensive guidelines to the students in advance by means of a brochure/handbook    ]'].value_counts(sort=1)
guidline_arr = guidlin_rating.to_numpy().tolist()
#print(guidline_arr)
print(guidlin_rating)

#rating of conducive learning environment
conducive_learning = df['Area of Evaluation [Department ensures a conducive learning environment]'].value_counts(sort=1)
conducive_arr = conducive_learning.to_numpy().tolist()
print(conducive_arr)

#rating of decision taken fairness
decision_fairness = df['Area of Evaluation [Academic decisions are taken with fairness and transparency]'].value_counts(sort=1)
decision_arr = decision_fairness.to_numpy().tolist()

#rating of academic calender maintance
calender = df['Area of Evaluation [	Academic calendar is maintained properly]'].value_counts(sort=1)
calender_arr = calender.to_numpy().tolist()

#rating of result publishing
result = df['Area of Evaluation [Results are published timely in compliance with the ordinance]'].value_counts(sort=1)
result_arr = result.to_numpy().tolist()

#rating of Opinion address
opinion = df['Area of Evaluation [Students� opinion regarding academic and extra-academic matters are addressed properly]'].value_counts(sort=1)
opinion_arr = opinion.to_numpy().tolist()

#rating of feedback applyed
feedback = df['Area of Evaluation [Student feedback process is in practice]'].value_counts(sort=1)
feedback_arr = feedback.to_numpy().tolist()

#rating of website
website = df['Area of Evaluation [Website is informative and updated properly]'].value_counts(sort=1)
website_arr = website.to_numpy().tolist()


#rating of curriculum load
curriculum = df['Area of Evaluation [Curriculum load is optimum and induces no pressure]'].value_counts(sort=1)
curriculum_arr = curriculum.to_numpy().tolist()


#putting data on object for home view
result = {
    'male'       : male,
    'female'     : female,
    'accuracy'   : accuracy * 100,
    'guidline'   : guidline_arr,
    'conducive'  : conducive_arr,
    'decision'   : decision_arr,
    'calender'   : calender_arr,
    'result'     : result_arr,
    'opinion'    : opinion_arr,
    'feedback'   : feedback_arr,
    'website'    : website_arr,
    'curriculum' : curriculum_arr
}

# Create your views here.
def home_view(request, *args, **kwargs):
    return render(request, 'home.html', result)

def about_view(request, *args, **kwargs):
    return render(request, 'about.html', {})

def contact_view(request, *args, **kwargs):
    return render(request, 'contact.html', {})

def developer_view(request, *args, **kwargs):
    return render(request, 'developer.html', {})