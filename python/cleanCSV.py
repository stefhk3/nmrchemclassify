import csv
rows = []
file =  open("../tox/Fathead_minnow_LC50_(96_hr)_Consensus2.csv", 'r')
csvreader = csv.reader(file,delimiter=",")
# header = next(csvreader)
rows.append(next(csvreader))
for row in csvreader:
    if  row[4] != "Structure can't be parsed":
        rows.append(row)
# print(header)
print(rows)
cleanedFile =  open("../tox/cleanedInputs/Cleaned_Fathead_minnow_LC50_(96_hr)_Consensus2.csv", 'w', newline='')
cleanedCSV = csv.writer(cleanedFile)
cleanedFile.writerows(rows)
