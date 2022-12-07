import csv

def main():
    f = open("Models.csv","r")
    Lines = f.readlines()
    count =0
    DataCsv = []
    name = ""
    f = open('Cleaned.csv','w')
    writer = csv.writer(f)
    dataset = -1
    for line in Lines:
        if count% 12 == 1:
            name = line
            dataset +=1
            if dataset > 6:
                dataset = 0
        #get loss accuracy and val loss and accuracy
        if ( count %2 ==1 and count %12 >2):
            print(line.strip().split(',')[0])
        if (count % 2 == 1) and count%12 > 4:
            s_line = line.strip()
            column = s_line.split(',')
            
            if count< 85:#if new trial
                tmpname = column[0] + "_" + name.strip()
                column.pop(0)
                column = list(map(lambda x: round(x,3),list(map(float,column))))
                column.insert(0,tmpname)
                DataCsv.append(column)
            else:
                column.pop(0)
                column = list(map(lambda x: round(x,3),list(map(float,column))))
                for i in range(len(column)-1):
                    DataCsv[(dataset*4)+int((count%12-4)/2-0.5)][i+1] += column[i]
                #DataCsv[(dataset*4)+int((count%12-4)/2-0.5)] += list(map(lambda x: round(x,3),list(map(float,column))))
        count +=1
    for Data in range(len(DataCsv)):
        try:
            tmpname = DataCsv[Data].pop(0)
            for i in range(len(DataCsv[Data])-1):
                DataCsv[Data][i]/=10
            DataCsv[Data] = list(map(lambda x: round(x,3),list(map(float,DataCsv[Data]))))
            DataCsv[Data].insert(0,tmpname)
        except:
            print("empty row")
    #print(DataCsv)
    #for data in DataCsv:  
    writer.writerows(DataCsv)
    f.close()
main()
