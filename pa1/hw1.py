# CAPP 30254
# PA1
# Santiago Larrain
# slarrain@uchicago.edu

import csv
import numpy as np
import matplotlib.pyplot as plt
import requests
import json

def import_data(filename):
    students = {}
    with open (filename, 'rU') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            students[row['ID']] = row
    return students

def lists_for_values(students):
    age = []
    gpa = []
    days_missed = []
    for id in students:
        if students[id]['Age'] != '':
            age.append(float(students[id]['Age']))
        if students[id]['GPA'] != '':
            gpa.append(float(students[id]['GPA']))
        if students[id]['Days_missed'] != '':
            days_missed.append(float(students[id]['Days_missed']))
    return age, gpa, days_missed

def summary(students):
    age, gpa, days_missed = lists_for_values(students)
    print 'AGE:'
    print '     Mean: '+str(np.mean(age))
    print '     Median: '+str(np.median(age))
    print '     Mode: '+str(np.bincount(age).argmax())
    print '     Standard Deviation: '+str(np.std(age))
    print ''
    print 'GPA:'
    print '     Mean: '+str(np.mean(gpa))
    print '     Median: '+str(np.median(gpa))
    print '     Mode: '+str(np.bincount(gpa).argmax())
    print '     Standard Deviation: '+str(np.std(gpa))
    print ''
    print 'Days Missed:'
    print '     Mean: '+str(np.mean(days_missed))
    print '     Median: '+str(np.median(days_missed))
    print '     Mode: '+str(np.bincount(days_missed).argmax())
    print '     Standard Deviation: '+str(np.std(days_missed))
    print ''

    #Create the correct bin variables for each histogram
    agebin = sorted(list(set(age)))
    agebin.append(max(agebin)+1)
    gpabin = sorted(list(set(gpa)))
    gpabin.append(max(gpabin)+1)
    daysbin = sorted(list(set(days_missed)))
    daysbin.append(max(daysbin)+1)

    hist_age, bins_age = np.histogram(age, bins=agebin)
    hist_gpa, bins_gpa = np.histogram(gpa, bins=gpabin)
    hist_days, bins_days = np.histogram(days_missed, bins=daysbin)

    # Plot Age Histogram
    plt.figure()
    plt.bar(bins_age[:-1], hist_age)
    plt.xlim(min(bins_age), max(bins_age))
    plt.xlabel("Age")
    plt.ylabel("Students")
    plt.title("Age Histogram")
    plt.savefig("Age_Histogram.png")

    #Plot GPA Histogram
    plt.figure()
    plt.bar(bins_gpa[:-1], hist_gpa)
    plt.xlim(min(bins_gpa), max(bins_gpa))
    plt.xlabel("GPA")
    plt.ylabel("Students")
    plt.title("GPA Histogram")
    plt.savefig("GPA_Histogram.png")

    #Plot Days Missing Histogram
    plt.figure()
    plt.bar(bins_days[:-1], hist_days)
    plt.xlim(min(bins_days), max(bins_days))
    plt.xlabel("Days missed")
    plt.ylabel("Students")
    plt.title("Days Missed Histogram")
    plt.savefig("Days_Missed_Histogram.png")


def missing_values (students):

    mv = {}
    for id in students:
        for item in students[id]:
            if students[id][item] == '':
                mv[item] = mv.get(item, 0) + 1
    print 'Missing Values:'
    for key, value in mv.items():
        print key, value

def define_gender(students):

    for id in students:
        if students[id]['Gender'] == '':
            print 'ID = '+id
            gender = fetch_gender(students[id]['First_name'])
            students[id]['Gender'] = gender
    create_file(students, '2')

def create_file(students, task):

    tasks = {'2': 'students_with_gender.csv', '3a': 'students_3a.csv',
            '3b': 'students_3b.csv',
            '3c': 'students_3c.csv'}
    keys = ['ID','First_name','Last_name','State', 'Gender','Age',
            'GPA','Days_missed','Graduated']
    with open (tasks[task], 'wb') as f:
        newfile = csv.writer(f)
        first = True
        for x in range(1, 1001):
            if first:
                newfile.writerow(keys)
                first = False
            newfile.writerow([students[str(x)][key] for key in keys])


def fetch_gender(name):
    req = requests.get("http://api.genderize.io?name=" + name)
    result = json.loads(req.text)
    return result['gender']

def fill_3a(students):

    age, gpa, days_missed = lists_for_values(students)
    students3a = dict(students)
    age_mean = int(np.mean(age))
    gpa_mean = int(np.mean(gpa))
    days_mean = int(np.mean(days_missed))
    for id in students3a:
        if students3a[id]['Age'] == '':
            students3a[id]['Age'] = age_mean
        if students3a[id]['GPA'] == '':
            students3a[id]['GPA'] = gpa_mean
        if students3a[id]['Days_missed'] == '':
            students3a[id]['Days_missed'] = days_mean
    create_file(students3a, '3a')

def fill_3b(students):

    age, gpa, days_missed = lists_for_values(students)
    students3b = dict(students)

    age_nograd = []
    gpa_nograd = []
    days_missed_nograd = []
    age_grad = []
    gpa_grad = []
    days_missed_grad = []
    for id in students3b:
        if students[id]['Age'] != '':
            if students[id]['Graduated'] == 'Yes':
                age_grad.append(float(students[id]['Age']))
            else:
                age_nograd.append(float(students[id]['Age']))
        if students[id]['GPA'] != '':
            if students[id]['Graduated'] == 'Yes':
                gpa_grad.append(float(students[id]['GPA']))
            else:
                gpa_nograd.append(float(students[id]['GPA']))
        if students[id]['Days_missed'] != '':
            if students[id]['Graduated'] == 'Yes':
                days_missed_grad.append(float(students[id]['Days_missed']))
            else:
                days_missed_nograd.append(float(students[id]['Days_missed']))

    mean_age_grad = int(np.mean(age_grad))
    mean_age_nograd = int(np.mean(age_nograd))
    mean_gpa_grad = int(np.mean(gpa_grad))
    mean_gpa_nograd = int(np.mean(gpa_nograd))
    mean_days_grad = int(np.mean(days_missed_grad))
    mean_days_nograd = int(np.mean(days_missed_nograd))

    for id in students3b:
        if students[id]['Age'] == '':
            if students[id]['Graduated'] == 'Yes':
                students3b[id]['Age'] = mean_age_grad
            else:
                students3b[id]['Age'] = mean_age_nograd
        if students[id]['GPA'] == '':
            if students[id]['Graduated'] == 'Yes':
                students3b[id]['GPA'] = mean_gpa_grad
            else:
                students3b[id]['GPA'] = mean_gpa_nograd
        if students[id]['Days_missed'] == '':
            if students[id]['Graduated'] == 'Yes':
                students3b[id]['Days_missed'] = mean_days_grad
            else:
                students3b[id]['Days_missed'] = mean_days_nograd

    create_file(students3b, '3b')


def best_filler(students, student, missing_val):
    '''
    This function tries to find a "best fit" by looking for people with similar
    demographics (keys list) according to the following keys:
    'Graduated', 'State', 'Gender', 'Age', 'GPA', 'Days_missed'
    If no one is found, we randomly drop one of the keys and try again.
    If we drop all the keys and still haven't found one, we just give him the
    values of the prior student.
    '''
    keys = ['Graduated', 'State', 'Gender', 'Age', 'GPA', 'Days_missed']
    keys.remove(missing_val)
    similar = []
    value_list = []

    while (len(similar)==0): #Iterate until we find at least one similar student
        for id in students:
            if students[id] != student: # Dont compare to himself
                count = 0 #Keep counts of how many similar atributes a given student has
                for key in keys:
                    if students[id][key] == student[key]:
                        count += 1
                if count == len(keys) and students[id][missing_val] != '':
                    #Append if the new student is similar on all keys and is not missing the missing value
                    similar.append(id)
        if len(keys) != 0:
            del keys[np.random.randint(len(keys))] # Deletes a random value of keys
        else: #If no best match was found
            return students[str(int(id)-1)][missing_val] #Just return the previous students value


    for id in similar:
        #if students[id][missing_val] != '':
        value_list.append(int(students[id][missing_val]))
    mean = int(np.mean(value_list)) #Calculate the mean of the similar students

    return mean



def fill_3c(students):
    '''
    It uses the best_filler function to try to find the best fit
    '''
    students3c = dict(students)

    for id in students3c:
        for data in ['Age', 'GPA', 'Days_missed']:
            if students3c[id][data] == '':
                students3c[id][data] = best_filler(students3c,
                                        students3c[id], data)
    create_file(students3c, '3c')


def do():

    filename_default = 'mock_student_data.csv'
    filename_gender = 'students_with_gender.csv'

    a = import_data(filename_default)

    summary(a)
    missing_values(a)

    # This function takes a couple of minutes to run, so you might want to
    # comment it out, specially since the outuput file has been submitted:
    # students_with_gender.csv
    define_gender(a)

    b = import_data(filename_gender)
    c = import_data(filename_gender)
    d = import_data(filename_gender)

    fill_3a(b)
    fill_3b(c)
    fill_3c(d)

if __name__ == "__main__":
    do()
