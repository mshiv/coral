import re
import os
myFiles = os.listdir()
myGauges = []
for f in myFiles:
    if re.search("gauge.*txt", str(f)) != None:
        myGauges.append(f)
myGauges.sort()
 
print(myGauges[:])
outputFile = open("Sorted.dat", "w")
outputFile.write("comment\n")
for f in range(len(myGauges[:])):
    theFile = open(str(myGauges[f]), "r")
    if theFile != None:
        rawData = []
         
        theContent = theFile.readlines()
        #theNum = len (theContent) - 3
        #outputFile.write("comment\n")
        #outputFile.write("bc" + str(f + 1) + "\n")
        #outputFile.write(str(theNum) + "\t\t" + "seconds" + "\n")
        for theLine in theContent[4:]:
            splittedLine = theLine.split()
            rawData.append((float(splittedLine[5]),float(splittedLine[1])))
        rawData.sort(key=lambda tup: tup[1])
       # rawData = rawData[:]
        outputFile.write("bc" + str(f + 1) + "\n")
        outputFile.write(str(len(rawData)) + "\t\t" + "seconds" + "\n")
        for t in rawData:
            outputFile.write(str(t[0]) + "\t" + str(t[1]) + "\t"  + "\n")
outputFile.close()
            
 


         
