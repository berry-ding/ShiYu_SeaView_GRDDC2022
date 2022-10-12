import tqdm
import os


parser = argparse.ArgumentParser(description='...')
parser.add_argument('file1', help='file1')
parser.add_argument('file2', help='file2')
parser.add_argument('output', help='outputfile')
args = parser.parse_args()

file1 = args.file1
file2 = args.file2
output = args.output

with open(file1, 'r') as f1:
    with open(file2, 'r') as f2:
        with open(output, 'a') as w:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
            line2_dict = {}
            for item in lines2:
                name, box = item.split(',')
                line2_dict[name] = box
            for idx in tqdm.tqdm(range(len(lines1))):
                name1, bboxs1 = lines1[idx].split(',')
                bboxs1 = bboxs1[:-1]
                if len(bboxs1) < 5:
                    bboxs = line2_dict[name1]
                elif len(line2_dict[name1]) < 5:
                    bboxs = bboxs1 + '\n'
                else:
                    bboxs = bboxs1 + line2_dict[name1]
                w.write(name1 + ',' + bboxs) 