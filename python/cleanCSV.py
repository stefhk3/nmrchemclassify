import csv


def cleanInputFile(fileName):
    readFrom = "../tox/"
    rows = []
    file =  open(readFrom+fileName, 'r')
    csvreader = csv.reader(file,delimiter=",")
    # header = next(csvreader)
    rows.append(next(csvreader))
    for row in csvreader:
        if  row[4] != "Structure can't be parsed":
            rows.append(row)
    # print(header)
    #print(rows)
    cleanedFile =  open(readFrom+"cleanedInputs/Cleaned_"+fileName, 'w', newline='')
    cleanedCSV = csv.writer(cleanedFile)
    cleanedCSV.writerows(rows)


cleanInputFile("Fathead_minnow_LC50_(96_hr)_Consensus2.csv")
