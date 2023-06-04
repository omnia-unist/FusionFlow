import os
import glob
import re
import copy
import pandas as pd

def replace_str(target):
    target = target.replace('\n', '')
    target = target.replace(',', '')
    return target

def find_value(arr, target, jumpto=1):
    try:
        num = replace_str(arr[arr.index(target)+jumpto])
    except:
        raise ValueError(
            f'{arr}, {target}, {arr[arr.index(target)+jumpto]}')
        num = 'NA'
    return num

io_time_pattern = re.compile(".* I/O_time:")
transform_pattern = re.compile("Transform .* END at_time")
process_time_pattern = re.compile("Load: .* Aug: .* End .*")
image_size_pattern = re.compile("image_size: .*")
logdir = "./log"
logdir_list = logdir.split('/')

output_file = f"test_aug.log"
logfile = logdir + "/" + output_file

col_name = ["Disk I/O time", "Buffer Cache I/O time"]
parsed_dir = "./io_parsed"
image_size_log = []
image_loglet = {}
image_size = ''
for line in open(logfile, 'r').readlines():
    if re.search(image_size_pattern, line) is not None:  # start log
        replace_txt = line.replace('\t', ' ')
        test = replace_txt.split(' ')
        info = list(filter(lambda x: x != "", test))
        if image_size != '':
            image_df = pd.DataFrame(image_size_log,
                                columns=col_name)
            os.makedirs(f'{parsed_dir}',exist_ok=True)
            image_df.to_csv(f'{parsed_dir}/{image_size}.csv', sep=',')
            image_size_log = []
        image_size = find_value(info, "image_size:")
    elif re.search(io_time_pattern, line) is not None:  # start log
        replace_txt = line.replace('\t', ' ')
        test = replace_txt.split(' ')
        info = list(filter(lambda x: x != "", test))
                
        io_time = find_value(info, "I/O_time:")
        if "Disk I/O time" in image_loglet:
            image_loglet["Buffer Cache I/O time"] = io_time
        else:
            image_loglet["Disk I/O time"] = io_time
    # elif re.search(transform_pattern, line) is not None:  # start log
    #     replace_txt = line.replace('\t', ' ')
    #     test = replace_txt.split(' ')
    #     info = list(filter(lambda x: x != "", test))
        
    #     transform_name = find_value(info, "Transform")
        
        
    #     if transform_name == '<__main__.RandAug':
    #         pass
    #     else:
    #         transform_name = transform_name.split('(')[0]
    #         aug_time = find_value(info, "at_time")
    #         image_loglet[transform_name] = aug_time
            
    #         if transform_name not in col_name:
    #             col_name.append(transform_name)
                
    #         if transform_name == 'Normalize':
    #             image_size_log.append(image_loglet)
    #             image_loglet = {}

#     elif re.search(process_time_pattern, line) is not None:  # start log
#         replace_txt = line.replace('\t', ' ')
#         test = replace_txt.split(' ')
#         info = list(filter(lambda x: x != "", test))
image_df = pd.DataFrame(image_size_log,
                    columns=col_name)
os.makedirs(f'{parsed_dir}',exist_ok=True)
image_df.to_csv(f'{parsed_dir}/{image_size}.csv', sep=',')

dataset_col_name = ["Image size (KB)"] + col_name

dir_names = glob.glob(f'{parsed_dir}/*.csv')

total_log = []
for logdir in dir_names:
    file_name = logdir.split('/')[-1]
    image_size = file_name.replace('.csv','')

    df = pd.read_csv(
        f"./{parsed_dir}/{file_name}", index_col=None)
    df = df*1000
    
    total_loglet = {"Image size (KB)":image_size}
    
    for aug in col_name:
        aug_df = df[aug].dropna() 
        total_loglet[aug] = f'{round(aug_df.mean(),2)}'
        
    total_log.append(total_loglet)

avg_df = pd.DataFrame(total_log,
                  columns=dataset_col_name)
avg_df.dropna().to_csv(f"./io_total_summary.csv",
                   sep=',', na_rep='NA')
