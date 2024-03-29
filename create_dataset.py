import numpy as np
import pandas as pd
import os
import sys
import csv
import random
random.seed(66)

# Split a list into folds
def split_list(x, folds=5):
    avg = len(x) // folds
    remainder = len(x) % folds
    return [x[i * avg + min(i, remainder):(i + 1) * avg + min(i + 1, remainder)] for i in range(folds)]

# Build a mapping
def get_subject_mapping(fn):
    with open(fn, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        data = [row for row in reader]

    header = data[0]
    data = data[1:]

    # Load column names
    props = set([])
    col_to_vals = [[]]
    for item in header[1:]:
        location, cluster = item.split('.')[:2]
        cluster = location + '/' + cluster

        prop = '.'.join(item.split('.')[2:])

        props.add(prop)

        col_to_vals.append({'cluster': cluster, 'prop': prop})

    # Load subject values
    subjects = {}
    for row in data:
        subject_id = row[0]
        col_num = 1
        subjects[subject_id] = {}
        for val in row[1:]:
            cluster = col_to_vals[col_num]['cluster']
            prop = col_to_vals[col_num]['prop']

            if 'NAN' in val:
                val = 0
            if cluster not in subjects[subject_id]:
                subjects[subject_id][cluster] = {prop: float(val)}
            else:
                subjects[subject_id][cluster][prop] = float(val)
            
            col_num += 1

    return subjects

def get_clusters(fn):
    with open(fn) as f:
        data = f.readlines()
    data = [x.strip('\n').replace('.', '/') for x in data]

    return sorted(data)

def save_data(x, out_fn):
    with open(out_fn, 'w', newline='') as file:
        writer = csv.writer(file)
        for key, value in x.items():
            writer.writerow([key] + value)


out_dir = input("Name of folder to save dataset to:")

# Make the output folder
if os.path.exists(out_dir):
    print('Folder already exists. Please try again with a non-existent folder name.')
    sys.exit()
os.mkdir(out_dir)

print("Creating dataset...")

# Load the requisite data
subjects = get_subject_mapping('HCP_n1065_allDWI_fiber_clusters.csv') 
target_clusters = get_clusters('clusters.csv')
feature_types = ['FA1', 'FA2', 'Num_Fibers', 'Num_Points', 'correct_trace1', 'correct_trace2']
stat_types = ['Min', 'Max', 'Median', 'Mean', 'Variance']

input_data = {}
for subject in subjects.keys():
    vector = []
    header = ['subject']
    for cluster in target_clusters:
        for feature in feature_types:
            if feature in ['Num_Fibers', 'Num_Points']:
                subject_data = subjects[subject][cluster][feature]
                vector.append(subject_data)
                header.append(cluster + '.' + feature)
            else:
                for stat_type in stat_types:
                    # Manually ignore trace1/trace2 variance, since that column is always 0
                    if feature in ['correct_trace1', 'correct_trace2'] and stat_type == 'Variance':
                        continue
                    subject_data = subjects[subject][cluster][feature + "." + stat_type]
                    vector.append(subject_data)
                    header.append(cluster + '.' + feature + '.' + stat_type)
    input_data[subject] = vector

# Save the processed data as a .csv with appropriate column headers
save_data(input_data, out_dir + '/subject_data.csv')
df = pd.read_csv(out_dir + '/subject_data.csv', header=None)
df.columns = header
df.to_csv(out_dir + '/subject_data.csv', index=False)

# Load the file again
with open(out_dir + '/subject_data.csv') as f:
    data = f.readlines()
subject_data = {}
for line in data[1:]:
    line = line.split(',')
    subject_data[line[0]] = np.array([float(x) for x in line[1:]])

# Create the folds
num_folds = 5
for i in range(num_folds):
    os.mkdir(out_dir + '/fold' + str(i))

# Get a list of subjects and sort them randomly
subjects = list(subject_data.keys())
random.shuffle(subjects)

# Split into num_folds folds
folds = split_list(subjects, folds=num_folds)

# Save the data to each fold
print('Generating...')
for i in range(num_folds):
    for subject in folds[i]:
        in_data = subject_data[subject]
        np.save(out_dir + '/fold' + str(i) + '/' + subject + '.npy', in_data)
    print("Fold %d done." % (i))
