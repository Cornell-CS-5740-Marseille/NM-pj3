import subprocess
import glob
def writeToFile(source, target):
    with open(source, "r") as readFile:
        with open(target, "w+") as writeFile:
            lines = readFile.readlines()
            for line in lines[:-1]:
                writeFile.write(line)
            lastLine = lines[-1]
            writeFile.close()
            readFile.close()
    return lastLine

def getAccuracy(sentence):
    sentence = sentence.split("Accuracy: ")
    accuracy = float(sentence[1]) * 100
    return accuracy

def outputBLEU(dir):
    print(dir)
    tasks = ["copy", "reverse", "sort"]
    for task in tasks:
        path = dir + "/logs/" + task + "/"
        files = path + "sameTraining_*"
        fileList = glob.glob(files)
        fileDict = {}
        for filePath in fileList:
            fileName = filePath.split("/")[-1]
            value = fileName.split("_")[-1].split(".")[0]
            type = fileName.split("_")[-2]
            if not value in fileDict:
                fileDict[value] = {}
            fileDict[value][type] = filePath
        # evaluation
        values = sorted(map(int, fileDict.keys()))
        for value in values:
            key = str(value)
            files = fileDict[key]
            last = writeToFile(files["reference"], "reference.txt")
            writeToFile(files["candidate"], "candidate.txt")
            # command = "perl multi-bleu.perl reference.txt < candidate.txt"
            command = "perl multi-bleu.perl " + files["reference"] + " < " + files["candidate"]
            output = subprocess.check_output(command, shell=True)[:-1]
            print("Task: ", task, "Value: ", value)
            print(output)
            print(last)
            print("%.2f" % getAccuracy(last))

# output = subprocess.check_output("perl multi-bleu.perl varyDictSize/logs/copy/reference_20.csv < varyDictSize/logs/copy/candidate_20.csv", shell=True)
# print(output)

# outputBLEU("varyDictSize")
# outputBLEU("varySequenceLength")
# outputBLEU("varySpecialSize")
outputBLEU("varyTrainingSize")