import os
import csv

def congate_csv(csv_dir,csv_prefix,final_csv_output):
    csv_file = open(final_csv_output,'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['image_name','Type_1','Type_2','Type_3'])
    for i in range(8):
        csvname = os.path.join(csv_dir,csv_prefix+"_%d"%i+".csv")
        csv_reader = csv.reader(open(csvname,'r'))
        for row in csv_reader:
            csv_writer.writerow(row)
    csv_file.close()

congate_csv('/mnt/lustre/dingyang/','cccc3','/mnt/lustre/dingyang/cccc_final.csv')
