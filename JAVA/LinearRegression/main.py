with open(r'train.csv', 'r', encoding="utf8") as inputfile:
    with open(r'trainc.csv', 'w', encoding="utf8") as outputfile:
        for line in inputfile:
            line = line.split('\t')
            outputfile.write(",".join(line).replace('Ł','L'))
    outputfile.close()
inputfile.close()