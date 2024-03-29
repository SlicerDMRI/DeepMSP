import csv
import sys
import os

def calculate_column_means(input_filename, desired_columns):
    column_sums = {col: 0.0 for col in desired_columns}
    column_counts = {col: 0 for col in desired_columns}
    with open(input_filename, 'r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            for col in desired_columns:
                if row[col].strip():
                    column_sums[col] += float(row[col])
                    column_counts[col] += 1
    column_means = {col: column_sums[col]/column_counts[col] for col in desired_columns if column_counts[col] > 0}
    return column_means

def extract_columns_with_mean(input_filename, output_filename, desired_columns):
    column_means = calculate_column_means(input_filename, desired_columns)
    with open(input_filename, 'r') as infile:
        reader = csv.DictReader(infile)
        fieldnames = ['Subject'] + desired_columns  # Always include the 'subject ID' column
        with open(output_filename, 'w', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                new_row = {'Subject': row['Subject']}
                for col in desired_columns:
                    if row[col].strip():
                        new_row[col] = row[col]
                    else:
                        new_row[col] = column_means.get(col, '')
                writer.writerow(new_row)


# Check if output folder already exists
if os.path.exists('./csvs'):
    print("'csvs' folder already exists.")
    sys.exit()
else:
    os.mkdir('./csvs')

names = ["Endurance_AgeAdj","GaitSpeed_Comp","Dexterity_AgeAdj","Strength_AgeAdj","PicSeq_AgeAdj","CardSort_AgeAdj","Flanker_AgeAdj","PMAT24_A_RTCR","ReadEng_AgeAdj","PicVocab_AgeAdj","ProcSpeed_AgeAdj","DDisc_AUC_200","VSPLOT_CRTE","SCPT_TPRT","IWRD_RTC","ListSort_AgeAdj"]
for name in names:
	input_filename = "S1200_demographics_Behavioral.csv"
	output_filename = "./csvs/" + name + ".csv"
	desired_columns = [name]

	extract_columns_with_mean(input_filename, output_filename, desired_columns)
