from datetime import datetime, timedelta
import os
import glob
import re
import copy
import pandas as pd
import multiprocessing
import psutil
CPU_NUM = psutil.cpu_count(logical = False)

trainTypeSet = [
    "SINGLE",
    # "DDP",
    # "DDP2GPU",
    # "DDP3GPU",
    # "DDP4GPU",
    # "DDP4GPUInter",
    # "DDP4GPUInterFULLTRACE",
    # "DDP4GPUInterFULLTRACE"
    "DDP6GPUInter",
    # "DDP6GPUInterFULLTRACEFETCH",
    # "DDP6GPUInterFULLTRACE",
    # "SINGLEFULLTRACE",
    # "DDP4GPUFULLTRACE",
    # "DDP2GPUFULLTRACE",
    # "DDP2GPUTwoSocket",
    # "DDP2GPUNoDiskSocket",
    # "DDP2GPUTwoSocket",
    # "DDP2GPUNoDiskSocket",
    # "DDPMANUAL",
    # "DDPFULLTRACE"
    # "DDPFULLTRACE2GPU"
]
trainType = "DDP4GPU"
# suffix_logdir = "GPU_Util"
suffix_logdir = ""
# suffix_logdir = "cpu"
logbase = f"./log/{suffix_logdir}"

logdir = f"{logbase}/{trainType}"
parsed_dir = f"dsanalyzer_parsed/{suffix_logdir}/{trainType}"


def redefine(tr_type):
    global trainType
    global logdir
    global parsed_dir
    trainType = tr_type
    logdir = f"{logbase}/{trainType}"
    parsed_dir = f"dsanalyzer_parsed/{suffix_logdir}/{trainType}"


logtypes = [
    # 'fordebug', 'fordebugnative', 'fordebugDecrease', 'fordebugIncrease', 'fordebugWorkerControll', 'fordebugprefetch1', 'fordebugprefetch3', 'fordebugprefetch4', 'fordebugprefetch8', 'fordebugprefetch16',
    # 'fordebugGlobalController','fordebugFIFOPolicyGlobalController', 'fordebugIntraBackFIFOPolicyGM', 'BackIntraIterGM',
    # 'fordebugWorkerControlFactor1', 'fordebugWorkerControlFactor2', 'fordebugWorkerControllx2idle', 'fordebugWorkerControllx3idle', 'fordebugBackworkerFIFOPolicyGlobalController', 'fordebugIntraAdaptiveBackFIFOPolicyGM',
    # 'fordebugIntraAdaptiveAggrBackFIFOPolicyGM', 'BackIntraIterGMQuartRun', 'MicroQuartRun', 'CPUPolicyGPU', 'CPUGPUNoWorkerStealing',
    
    'origin_main_pin', 'baseline', 'CPUPolicy', 'CPUPolicyGPU', 'CPUPolicyGPUDirectAggr', 'dalibaseline', 'tfdata', 'GPUonly', 'GPUonlyNaive', 'CPUPolicyGPUDirectAggrOffload', 'CPUPolicyGPUDirectAggrOffloadSingleSamp', 
    'CPUPolicyGPUDirectAggrOffloadRefurbish', 'CPUPolicyGPUDirectAggrOffloadRefurbish25', 'CPUPolicyGPUDirectAggrOffloadRefurbish50', 'CPUPolicyGPUDirectAggrOffloadRefurbish100',
    'loadNtrain', 'fordebug'
    
    # 'train','train_only', 'train_onlyPersistent','trainNoAMP',
    # 'loadNtrain', 'loadNtrainPWPMPin', 'forMainonly', 
    # 'loadNtrainPersistent', 'loadNtrainPWPMPin','loadNtrainPWPMPinNoAMP',
    # #'prepNloadNtrain', 'prepNloadN trainPersistent','prepNloadNtrainPersistentOverwriteSampler',
    # #'prepNloadNtrainPersistentSyntheticCopy','prepNloadNtrainPersistentNodecodeCopy','prepNloadNtrainPersistentWorkerPinOverwriteSampler','prepNloadNtrainPinmemoryPinOverwriteSampler',
    # 'prepNloadNtrainPMPWPinOverwriteSampler','prepNloadNtrainPMPWPinOverwriteSamplerNoAMP', 'prepNloadNtrainPMPWPinOverwriteSamplerTwoGPU', 'prepNloadNtrainPMPWPinOverwriteSamplerTwoGPUdiffsocket',
    # 'baseline', 'IO', 'Refurbish', 'Micro', 'FIFOPolicy', 'BackFIFOPolicy', 'CPUPolicy',
    # '1thresBackFIFOPolicy', 'BackFIFOPolicydiffsocket', 'TwoGPUdiffsocket', 'FourGPUdiffsocket', 'AdaptiveBackInterFirstGM', 
    # 'HalfRun', 'QuartRun', 'CPUPolicyHalfRun', 'CPUPolicyQuartRun',
    # 'AdaptiveAggrBackGMQuartRun',
    # 'TwoGPU', 'NoPinCPU',
    # 'NoAMP',
    # 'fetchN',
    #'fetchNNoAMP', 'fetchNTwoGPU','fetchNTwoGPUdiffSocket'
]
# "prepNloadNtrainCustom", "prepNloadNtrainNoPin", "prepNloadNtrainSyncStart", 'prepNloadNtrainPersistentGarbage'
common_term = {
    '1024batch': 0.113,
    '1024batch10': 1.1,
    'size2': 2,
    'size3': 3,
    'size5': 5,
    'size10': 10,
    'size20': 20,
    'size40': 40,
    'size80': 80,
    'size128': 128,
    'size160': 160,
    'size320': 320,
    'size512': 512,
    'size80iter': 2.2,
    'Imagenet': 147,
    'imagenet': 147,
    'openimage': 517,
    'synth_inat': 146,
    'sampled_openimage': 52,
    'winter21whole': 1100,
    'default': 'Default',
    'randaugment': 'RandAugment',
    'autoaugment': 'AutoAugment',
    'randaugmentnorandom': 'RandAugment+No Rand',
    'augmix': 'ResNet+Augmix',
    'norandom': 'RN No Rand',
    'none': 'None',
    'noresizecrop': 'NoResizeCrop',
    'noresize': 'NoResize',
    'fliponlynorand': 'Flip and Tensor',
    'tensoronly': 'Tensor only',
    
}

term = {
    'fordebug': 'LoadOnlyMiniBatch',
    'fordebugnative': 'LoadOnlyNative',
    'fordebugprefetch1': 'LoadOnlyMiniBatchPrefetch1',
    'fordebugprefetch3': 'LoadOnlyMiniBatchPrefetch3',
    'fordebugprefetch4': 'LoadOnlyMiniBatchPrefetch4',
    'fordebugprefetch8': 'LoadOnlyMiniBatchPrefetch8',
    'fordebugprefetch16': 'LoadOnlyMiniBatchPrefetch16',
    'fordebugDecrease': 'LoadOnlyDecreaseWorker',
    'fordebugIncrease': 'LoadOnlyIncreaseWorker',
    'fordebugBackworkerFIFOPolicyGlobalController': 'LoadOnlyBackgroundFIFO',
    'fordebugWorkerControlFactor1': 'LoadOnlyWorkerControl1', 
    'fordebugWorkerControlFactor2': 'LoadOnlyWorkerControl2',
    'fordebugWorkerControll': 'LoadOnlyWorkerStale',
    'fordebugWorkerControllx2idle': 'LoadOnlyWorkerStalex2idle',
    'fordebugWorkerControllx3idle': 'LoadOnlyWorkerStalex3idle',
    'fordebugGlobalController': 'LoadOnlySchedulerInfoCommunication',
    'fordebugFIFOPolicyGlobalController': 'LoadOnlyFIFOPolicy',
    'fordebugIntraBackFIFOPolicyGM': 'LoadOnlyIntraBackFIFOPolicy',
    'fordebugIntraAdaptiveBackFIFOPolicyGM': 'LoadOnlyIntraBackAdaptiveFIFOPolicy',
    'fordebugIntraAdaptiveAggrBackFIFOPolicyGM': 'LoadOnlyIntraAggrBackAdaptiveFIFOPolicy',
    'forMainonly': 'LoadOnlyOrigin',
    'train': '⑤',
    'trainNoAMP': '⑤NoAMP',
    'train_only': '⑤',
    'trainPersistent': '⑤Per',
    'train_onlyPersistent': '⑤Per',
    'loadNtrain': '④+⑤',
    'loadNtrainPersistent': '④+⑤Persistent',
    'loadNtrainPWPMPin': '④+⑤PWPM',
    'loadNtrainPWPMPinNoAMP': '④+⑤NoAMP',
    'prepNloadNtrain': '③+④+⑤',
    'prepNloadNtrainNoPin': '③+④+⑤noPinmemory',
    'prepNloadNtrainSyncStart': '③+④+⑤SyncStart',
    'prepNloadNtrainCustom': '③+④+⑤Custom',
    'prepNloadNtrainPersistent': '③+④+⑤Persistent',
    'prepNloadNtrainPersistentGarbage': '③+④+⑤PersistentGarbage',
    'prepNloadNtrainPersistentSyntheticCopy': '③+④+⑤Synthetic',
    'prepNloadNtrainPersistentNodecodeCopy': '③+④+⑤Nodecode',
    'prepNloadNtrainPersistentOverwriteSampler': '③+④+⑤Overwrite',
    'prepNloadNtrainPersistentWorkerPinOverwriteSampler': '③+④+⑤Persistent',
    'prepNloadNtrainPinmemoryPinOverwriteSampler': '③+④+⑤',
    'prepNloadNtrainPMPWPinOverwriteSampler': '③+④+⑤PWPM',
    'prepNloadNtrainPMPWPinOverwriteSamplerNoAMP': '③+④+⑤NoAMP',
    'prepNloadNtrainPMPWPinOverwriteSamplerTwoGPU': '③+④+⑤Sckt0',
    'prepNloadNtrainPMPWPinOverwriteSamplerTwoGPUdiffsocket': '③+④+⑤Sckt1',
    'baseline': 'Train+baseline',
    'origin_main_pin': 'Load+pin',
    'IO': 'Train+IO',
    'NoPinCPU': 'Train+NoPin',
    'TwoGPU': 'Train+2GPU2Sckt',
    'NoAMP': 'Train+NoAMP',
    'Micro': 'Train+Micro',
    'MicroQuartRun': 'Train+MicroQuartRun',
    'FIFOPolicy': 'Train+FIFOPolicy',
    'BackFIFOPolicy': 'Train+BackgroundFIFOPolicy',
    '1thresBackFIFOPolicy': 'Train++thres1BackgroundFIFOPolicy',
    'TwoGPUdiffsocket': 'Train+Sckt1',
    'BackIntraIterGM': 'Train+IntraBackFIFOPolicy',
    'BackIntraIterGMQuartRun': 'Train+IntraBackFIFOPolicyQuartRun',
    'TwoGPUdiffSocket': 'Train+Baseline',
    'Refurbish': 'Train+ImitateRefurbish',
    'FourGPUdiffsocket': 'Train+FourGPUdiffsocket',
    'BackFIFOPolicydiffsocket': 'Train+BackgroundFIFOPolicy2Socket',
    'CPUPolicy': 'Train+CPUPolicy',
    'GPUonlyNaive': 'Train+GPUonlyNaive',
    'dalibaseline': 'Train+DALI',
    'tfdata': 'Train+tensorflow',
    'HalfRun': 'Train+Half',
    'QuartRun': 'Train+Quarter',
    'CPUPolicyHalfRun': 'Train+AdaptiveBackIntraInterGMHalf',
    'CPUPolicyQuartRun': 'Train+AdaptiveBackIntraInterGMQuarter',
    'AdaptiveAggrBackGMQuartRun': 'Train+AdaptiveBackAggrGMQuarter',
    'AdaptiveBackInterFirstGM': 'Train+AdaptiveBackInterFirstGM',
    'CPUPolicyGPU': 'Train+CPUPolicyGPU',
    'GPUonly': 'Train+GPU',
    'CPUPolicyGPUDirectAggr': 'Train+CPUPolicyGPUDirectAggr',
    'CPUGPUNoWorkerStealing': 'Train+CPUGPUNoWorkerStealing',
    'CPUPolicyGPUDirectAggrOffload': 'Train+CPUPolicyGPUDirectAggrOffload',
    'CPUPolicyGPUDirectAggrOffloadSingleSamp': 'Train+CPUPolicyGPUDirectAggrOffloadSingleSamp',
    'CPUPolicyGPUDirectAggrOffloadRefurbish': 'Train+CPUPolicyGPUDirectAggrOffloadRefurbish',
    'CPUPolicyGPUDirectAggrOffloadRefurbish25': 'Train+CPUPolicyGPUDirectAggrOffloadRefurbish25',
    'CPUPolicyGPUDirectAggrOffloadRefurbish50': 'Train+CPUPolicyGPUDirectAggrOffloadRefurbish50',
    'CPUPolicyGPUDirectAggrOffloadRefurbish100': 'Train+CPUPolicyGPUDirectAggrOffloadRefurbish100',
    # 'fetchN': '①+Train+',
    # 'fetchNNoAMP': '①+Train+NoAMP',
    # 'fetchNTwoGPU': '①+Train+Sckt0',
}

figterm = {}

for key in term:
    figterm[key] = term[key].replace('①', '1').replace(
        '②', '2').replace('③', '3').replace('④', '4').replace('⑤', '5')

term.update(common_term)
figterm.update(common_term)


class baseParser():
    def __init__(self, suffix_dir=""):
        self.suffix_dir = suffix_dir
        self.parsed_dir = parsed_dir + self.suffix_dir
        self.breakdown_col_name = []

    def set_dir(self):
        self.parsed_dir = parsed_dir + self.suffix_dir
        os.makedirs(self.parsed_dir, exist_ok=True)

        if trainType.find("2GPU") != -1:
            self.gpu_num = 2
        elif trainType.find("3GPU") != -1:
            self.gpu_num = 3
        elif trainType.find("4GPU") != -1:
            self.gpu_num = 4
        elif trainType.find("6GPU") != -1:
            self.gpu_num = 6
        elif trainType.find("DDP") != -1:
            self.gpu_num = 8
        else:
            self.gpu_num = 1
        self.simple_dir_pattern = logdir+'/{}'

        self.dir_names = self.get_dirname()
        print(self.dir_names)

    def get_dirname(self):
        dir_names = {}
        for logtype in logtypes:
            dir_names[logtype] = glob.glob(
                self.simple_dir_pattern.format(logtype)+f"/*/*/*/*/*/*/*/")

        return dir_names

    def replace_str(self, target):
        target = target.replace('\n', '')
        target = target.replace(',', '')
        return target

    def find_value(self, arr, target, jumpto=1):
        try:
            num = self.replace_str(arr[arr.index(target)+jumpto])
        except:
            raise ValueError(
                f'{arr}, {target}, {jumpto}')
            num = 'NA'
        return num

    def parser(self):
        raise NotImplementedError

class newBaseParser(baseParser):
    def __init__(self, suffix_dir=""):
        super(newBaseParser, self).__init__(suffix_dir)
        
    def parser(self):
        for tr_type in trainTypeSet:
            redefine(tr_type)
            self.set_dir()
            for logtype in logtypes:
                for logdir in self.dir_names[logtype]:
                    if not os.path.exists(logdir) or not os.path.isfile(logdir + "/" + self.output_file):
                        continue
                    logdir_list = logdir.split('/')

                    dataset = self.find_value(logdir_list, logtype)
                    aug = self.find_value(logdir_list, dataset)
                    model = self.find_value(logdir_list, aug)
                    epoch = self.find_value(logdir_list, model)
                    batchsize = self.find_value(logdir_list, epoch)
                    worker = self.find_value(logdir_list, batchsize)
                    thread = self.find_value(logdir_list, worker)
                    model = model.replace("_", "")
                    
                    epoch_num = epoch.replace("epoch", "")
                    batchsize_num = batchsize.replace("b", "")
                    worker_num = worker.replace("worker", "")
                    parse_filename = f"{logtype}_{dataset}_{aug}_{model}_{epoch}_{batchsize}_{worker}_{thread}"
                    logfile = logdir + "/" + self.output_file
                    
                    parse_filename, breakdown_parsed_log = self.parse_line(parse_filename, logfile)

                    breakdown_df = pd.DataFrame(breakdown_parsed_log,
                                                columns=self.breakdown_col_name)
                    breakdown_df.dropna().to_csv(self.parsed_dir+"/"+parse_filename +
                                                 ".csv", sep=',', na_rep='NA')
    
    def parse_line(self, parse_filename, logfile):
        raise NotImplementedError
    
class gpuParser(newBaseParser):
    def __init__(self):
        super(gpuParser, self).__init__("/gpu_util")
        self.breakdown_col_name = None
        self.output_file = "gpu_log.csv"
        os.makedirs(self.parsed_dir, exist_ok=True)
        
    def parse_line(self, parse_filename, logfile):
        self.gpu_num = 4
        df = pd.read_csv(logfile, header=None, names=["timestamp","utilization"], comment="t")

        breakdown_parsed_log = pd.DataFrame()
        breakdown_parsed_log["timestamp"] = df["timestamp"][::self.gpu_num].reset_index(drop=True)
        for i in range(self.gpu_num):
            breakdown_parsed_log[f"GPU{i} Utilization(%)"] = df["utilization"][i::self.gpu_num].reset_index(drop=True)
            
        return parse_filename, breakdown_parsed_log
    
    def total_summary(self):
        for tr_type in trainTypeSet:
            redefine(tr_type)
            self.set_dir()
            dataset_col_name = ["Log type", "Dataset size(GB)", "Model", "Augmentation", "Worker", "Worker batch size", "Epoch", "Batch size"]
            diff_col_name = ["Epoch","Step"]
            yaxis = []
            for i in range(self.gpu_num):
                dataset_col_name.append(f"GPU{i} avg utilization")
                diff_col_name.append(f"GPU{i} Utilization(%)")
                yaxis.append(f"GPU{i} Utilization(%)")
            total_log = []
            for logtype in logtypes:
                for logdir in self.dir_names[logtype]:
                    if not os.path.exists(logdir) or not os.path.isfile(logdir + "/" + self.output_file):
                        continue
                    logdir_list = logdir.split('/')
                    # print(logdir_list)
                    dataset = self.find_value(logdir_list, logtype)
                    aug = self.find_value(logdir_list, dataset)
                    model = self.find_value(logdir_list, aug)
                    epoch = self.find_value(logdir_list, model)
                    batchsize = self.find_value(logdir_list, epoch)
                    worker = self.find_value(logdir_list, batchsize) 
                    thread = self.find_value(logdir_list, worker)
                    model = model.replace("_", "")
                    # print(batchsize)
                    epoch_num = int(epoch.replace("epoch", ""))
                    batchsize_num = int(batchsize.replace("b", ""))
                    single_batchsize_num = batchsize_num/self.gpu_num
                    worker_num = int(worker.replace("worker", ""))
                    thread_num = float(thread.replace("thread", ""))

                    parse_filename = f"{logtype}_{dataset}_{aug}_{model}_{epoch}_{batchsize}_{worker}_{thread}"
                    df = pd.read_csv(
                        f"./{self.parsed_dir}/{parse_filename}.csv", index_col=None)
                    df["timestamp"] = pd.to_datetime(df["timestamp"], format='%Y/%m/%d %H:%M:%S.%f', errors='ignore')
                   
                    for y in yaxis:
                        df[y] = list(map(lambda x: x[:-1], df[y].values))
                        df[y] = df[y].astype(float, True)
                        
                    perf_df = pd.read_csv(
                        f"./{parsed_dir}/{parse_filename}.csv", index_col=None)
                    new_df = pd.DataFrame()
                    gpu_log = []

                    perf_df["Time"] = pd.to_datetime(perf_df["Time"], format='%Y-%m-%d %H:%M:%S.%f', errors='ignore')
                    for i in range(self.gpu_num):
                        new_df["START time"] = perf_df["Time"][:-1]
                        new_df["END time"] = perf_df["Time"][1:].reset_index()["Time"]
                    new_df["START time"] = pd.to_datetime(new_df["START time"], format='%Y-%m-%d %H:%M:%S.%f', errors='ignore')
                    new_df["END time"] = pd.to_datetime(new_df["END time"], format='%Y-%m-%d %H:%M:%S.%f', errors='ignore')
                    # print(new_df)
                    
                    diff_log = []
                    count = 0
                    epoch = 0
                    max_step=perf_df["Step"].max()
                    
                    for index, row in new_df.iterrows():
                        newrow = [epoch,count]
                        iter_df = df[(row["END time"] < df["timestamp"]) & (df["timestamp"] < row["START time"])]
                        for gpu_id in range(self.gpu_num):
                            newrow.append(iter_df[f"GPU{gpu_id} Utilization(%)"].mean())
                        count+=1
                        if count > max_step:
                            epoch += 1
                            count = 0
                        diff_log.append(newrow)

                    diff_df=pd.DataFrame(diff_log, columns=diff_col_name)
                    total_loglet = [term[logtype], term[dataset], model, term[aug], worker_num, thread_num, epoch_num, batchsize_num]
                    diff_df.to_csv(self.parsed_dir+f"/iter_util_summary.csv",
                                   sep=',', na_rep='NA')
                    
                    for gpu_id in range(self.gpu_num):
                        total_loglet.append(diff_df[f"GPU{gpu_id} Utilization(%)"].mean())
                    
                    # print(total_loglet)
                    total_loglet.extend(gpu_log)
                    total_log.append(total_loglet)

            avg_df = pd.DataFrame(total_log,
                                  columns=dataset_col_name)
            avg_df.dropna().to_csv(self.parsed_dir+f"/gpu_util_analyzer_total_summary.csv",
                                   sep=',', na_rep='NA')
            
            
class perfParser(baseParser):
    def __init__(self):
        super(perfParser, self).__init__()
        self.breakdown_col_name = [
            "Time", "GPU", "Epoch", "Step", "Iteration time (sec)", "Training stall time (sec)", "Throughput (image/sec)"]
        self.output_file = "output.log"
        self.parsed_dir = self.parsed_dir
        os.makedirs(self.parsed_dir, exist_ok=True)

    def parser(self):
        for tr_type in trainTypeSet:
            redefine(tr_type)
            self.set_dir()
            for logtype in logtypes:
                for logdir in self.dir_names[logtype]:
                    if not os.path.exists(logdir) or not os.path.isfile(logdir + "/" + self.output_file):
                                            continue
                    logdir_list = logdir.split('/')

                    dataset = self.find_value(logdir_list, logtype)
                    aug = self.find_value(logdir_list, dataset)
                    model = self.find_value(logdir_list, aug)
                    epoch = self.find_value(logdir_list, model)
                    batchsize = self.find_value(logdir_list, epoch)
                    worker = self.find_value(logdir_list, batchsize)
                    thread = self.find_value(logdir_list, worker)
                    model = model.replace("_", "")
                                               

                    epoch_num = epoch.replace("epoch", "")
                    batchsize_num = batchsize.replace("b", "")
                    worker_num = worker.replace("worker", "")

                    parse_filename = f"{logtype}_{dataset}_{aug}_{model}_{epoch}_{batchsize}_{worker}_{thread}"

                    logfile = logdir + "/" + self.output_file
                    breakdown_parsed_log = []

                    # Parse line one by one
                    for line in open(logfile, 'r').readlines():
                        if line.find(f'Epoch:') != -1:  # start log
                            replace_txt = line.replace('\t', ' ')
                            test = replace_txt.split(' ')
                            info = list(filter(lambda x: x != "", test))
                            # print(info)
                            try:
                                time_str1 = self.find_value(info, "GPU:", -2)
                                time_str2 = self.find_value(info, "GPU:", -1)
                                time_str = f"{time_str1} {time_str2}"
                                time = datetime.strptime(
                                    time_str, '%Y-%m-%d %H:%M:%S.%f')
                            except:
                                time = "nan"

                            try:
                                gpu_id = int(self.find_value(info, "GPU:"))
                            except:
                                gpu_id = "nan"

                            epochNstep = self.find_value(info, "Epoch:")
                            epoch = epochNstep.split(']')[0].replace('[', '')
                            if epochNstep.find('/') != -1:
                                step = epochNstep.split('[')[2].split('/')[0]
                            else:
                                step = self.find_value(
                                    info, "Epoch:", 2).split('/')[0]

                            # epoch avg: total n iter
                            iter_time = self.find_value(info, "Time")
                            data_time = self.find_value(
                                info, "Data")  # data time
                            throughput = self.find_value(
                                info, "Throughput")  # throughput avg
                            cur_log_info = [time, gpu_id, epoch, step,
                                            iter_time, data_time, throughput]
                            # print(cur_log_info)
                            # return
                            breakdown_parsed_log.append(cur_log_info)
                        else:
                            pass

                    breakdown_df = pd.DataFrame(breakdown_parsed_log,
                                                columns=self.breakdown_col_name)
                    breakdown_df.dropna().to_csv(self.parsed_dir+"/"+parse_filename +
                                                 ".csv", sep=',', na_rep='NA')

    def total_summary(self):
        for tr_type in trainTypeSet:
            redefine(tr_type)
            self.set_dir()
            dataset_col_name = ["Log type", "Dataset size(GB)", "Model", "Augmentation", "Worker", "Worker batch size", "Epoch", "Batch size", "Avg iteration time (sec)",
                                "Avg training stall time (sec)", "Avg throughput (images/sec)", "Avg processed size (MB/sec)"]#, "Avg filtered epoch time (sec)", "Avg epoch time (sec)"]
            for i in range(self.gpu_num):
                dataset_col_name.append(f"GPU{i} avg iteration time (sec)")
                dataset_col_name.append(
                    f"GPU{i} avg training stall time (sec)")
            total_log = []
            for logtype in logtypes:
                for logdir in self.dir_names[logtype]:
                    if not os.path.exists(logdir) or not os.path.isfile(logdir + "/" + self.output_file):
                        continue
                    logdir_list = logdir.split('/')
                    print(logdir_list)
                    dataset = self.find_value(logdir_list, logtype)
                    aug = self.find_value(logdir_list, dataset)
                    model = self.find_value(logdir_list, aug)
                    epoch = self.find_value(logdir_list, model)
                    batchsize = self.find_value(logdir_list, epoch)
                    worker = self.find_value(logdir_list, batchsize) 
                    thread = self.find_value(logdir_list, worker)
                    model = model.replace("_", "")
                    # print(batchsize)
                    epoch_num = int(epoch.replace("epoch", ""))
                    batchsize_num = int(batchsize.replace("b", ""))
                    single_batchsize_num = batchsize_num/self.gpu_num
                    worker_num = int(worker.replace("worker", ""))
                    thread_num = float(thread.replace("thread", ""))

                    parse_filename = f"{logtype}_{dataset}_{aug}_{model}_{epoch}_{batchsize}_{worker}_{thread}"
                    df = pd.read_csv(
                        f"./{self.parsed_dir}/{parse_filename}.csv", index_col=None)
                    # print(parse_filename)
                    # print(df)
                    # Fliter out
                    # df = df[df["Epoch"] > 0]
                    # df = df[df["Epoch"] < (epoch_num-1)]

                    gpu_log = []

                    for i in range(self.gpu_num):
                        gpu_df = df[df["GPU"] == i].dropna()
                        gpu_epoch_time = f'{round(gpu_df["Iteration time (sec)"].mean(),4)}±{round(gpu_df["Iteration time (sec)"].std(),4)}'
                        gpu_log.append(gpu_epoch_time)
                        gpu_stall_time = f'{round(gpu_df["Training stall time (sec)"].mean(),4)}±{round(gpu_df["Training stall time (sec)"].std(),4)}'
                        gpu_log.append(gpu_stall_time)

                    # avg_epoch_time = round(df["Iteration time (sec)"].astype(float).sum()/(int(epoch_num)-1), 2)

                    df = df[df["Step"] > 10]
                    df = df[df["Step"] < (df["Step"].max() - 10)]
                    # filtered_avg_epoch_time = round(
                    #     df["Iteration time (sec)"].astype(float).sum()/(int(epoch_num)-1), 2)
                    # print(df.describe())
                    # print("\n")
                    # print(parse_filename, df["Iteration time (sec)"])
                    iter_origin = df["Iteration time (sec)"].mean()

                    # print(iter_origin)

                    iter_avg = f'{round(iter_origin,4)}±{round(df["Iteration time (sec)"].std(),4)}'
                    data_avg = f'{round(df["Training stall time (sec)"].mean(),4)}±{round(df["Training stall time (sec)"].std(),4)}'
                    throughput_origin = single_batchsize_num/iter_origin
                    throughput_avg = round(throughput_origin, 4)
                    # # [Note] :
                    # # Deprecated throughput avg,
                    # # theoretically wrong in mathematic, Please check below link
                    # # https://fxloader.com/inverse_of_an_average_compared_to_averages_of_inverses/
                    # throughput_avg = f'{round(df["Throughput (image/sec)"].mean(),2)}±{round(df["Throughput (image/sec)"].std(),4)}'

                    # Hard coded as imagenet avg size (MB)
                    processed_data_avg = throughput_origin * 105.53 / 1024
                    
                    total_loglet = [term[logtype], term[dataset], model, term[aug], worker_num, thread_num, epoch_num, batchsize_num,
                                    iter_avg, data_avg, throughput_avg, processed_data_avg]
                    # print(total_loglet)
                    total_loglet.extend(gpu_log)
                    total_log.append(total_loglet)

            avg_df = pd.DataFrame(total_log,
                                  columns=dataset_col_name)
            avg_df.dropna().to_csv(self.parsed_dir+f"/df_analyzer_total_summary.csv",
                                   sep=',', na_rep='NA')

    def total_median_summary(self):
        for tr_type in trainTypeSet:
            redefine(tr_type)
            self.set_dir()
            dataset_col_name = ["Log type", "Dataset size(GB)", "Model", "Augmentation", "Worker", "Worker batch size", "Epoch", "Batch size", "Avg iteration time (sec)", "Avg training stall time (sec)", "Avg throughput (images/sec)", "Avg processed size (MB/sec)", "Avg filtered epoch time (sec)", "Avg epoch time (sec)",
                                "Median iteration time (sec)", "Median training stall time (sec)", "Median throughput (images/sec)"]
            total_log = []
            for logtype in logtypes:
                for logdir in self.dir_names[logtype]:
                    if not os.path.exists(logdir) or not os.path.isfile(logdir + "/" + self.output_file):
                        continue
                    logdir_list = logdir.split('/')

                    dataset = self.find_value(logdir_list, logtype)
                    aug = self.find_value(logdir_list, dataset)
                    model = self.find_value(logdir_list, aug)
                    epoch = self.find_value(logdir_list, model)
                    batchsize = self.find_value(logdir_list, epoch)
                    worker = self.find_value(logdir_list, batchsize)
                    thread = self.find_value(logdir_list, worker)
                    model = model.replace("_", "")

                    epoch_num = int(epoch.replace("epoch", ""))
                    batchsize_num = int(batchsize.replace("b", ""))
                    worker_num = int(worker.replace("worker", ""))
                    thread_num = float(thread.replace("thread", ""))

                    parse_filename = f"{logtype}_{dataset}_{aug}_{model}_{epoch}_{batchsize}_{worker}_{thread}"
                    df = pd.read_csv(
                        f"./{self.parsed_dir}/{parse_filename}.csv", index_col=None)
                    df = df[df["Step"] >  10]
                    # df = df[df["Epoch"] > 0]
                    # df = df[df["Epoch"] < (epoch_num-1)]

        #                 print(parse_filename, df["Iteration time (sec)"])
                    iter_avg = df["Iteration time (sec)"].median()
                    data_avg = df["Training stall time (sec)"].median()
                    throughput_avg = df["Throughput (image/sec)"].median()

                    total_log.append([term[logtype], term[dataset], model, term[aug], worker_num, thread_num,
                                      epoch_num, batchsize_num, iter_avg, data_avg, throughput_avg])

                avg_df = pd.DataFrame(total_log,
                                      columns=dataset_col_name)
                avg_df.dropna().to_csv(
                    parsed_dir+f"/df_analyzer_total_median_summary.csv", sep=',', na_rep='NA')
                
    def delay_summary(self):
        for tr_type in trainTypeSet:
            redefine(tr_type)
            self.set_dir()
            delay_col_name = ["Log type", "Dataset size(GB)", "Model", "Augmentation", "Worker", "Worker batch size", "Epoch", "Batch size",
                              "Avg delayed time (sec)"]
            dataset_gpu_name = [
                "Log type", "Dataset size(GB)", "Model", "Augmentation", "Worker", "Worker batch size", "Epoch", "Batch size"]

            for i in range(self.gpu_num):
                dataset_gpu_name.append(
                    f"GPU{i} batchSize * Avg Get target (sec)")
                dataset_gpu_name.append(
                    f"GPU{i} batchSize * Avg Fetch (sec)")
                dataset_gpu_name.append(
                    f"GPU{i} batchSize * Avg Decode (sec)")
                dataset_gpu_name.append(
                    f"GPU{i} batchSize * Avg Get image (sec)")
                dataset_gpu_name.append(f"GPU{i} batchSize * Avg Load (sec)")
                dataset_gpu_name.append(
                    f"GPU{i} batchSize * Avg Read and Decode (sec)")
                dataset_gpu_name.append(
                    f"GPU{i} batchSize * Avg Augmentation (sec)")
                dataset_gpu_name.append(f"GPU{i} batchSize * Avg Batch (sec)")

            total_delay_log = []
            for logtype in logtypes:
                for logdir in self.dir_names[logtype]:
                    if not os.path.exists(logdir) or not os.path.isfile(logdir + "/" + self.output_file):
                        continue
                    logdir_list = logdir.split('/')
                    # print(logdir_list)
                    dataset = self.find_value(logdir_list, logtype)
                    aug = self.find_value(logdir_list, dataset)
                    model = self.find_value(logdir_list, aug)
                    epoch = self.find_value(logdir_list, model)
                    batchsize = self.find_value(logdir_list, epoch)
                    worker = self.find_value(logdir_list, batchsize)
                    thread = self.find_value(logdir_list, worker)
                    model = model.replace("_", "")
                    # print(batchsize)
                    epoch_num = int(epoch.replace("epoch", ""))
                    batchsize_num = int(batchsize.replace("b", ""))
                    single_batchsize_num = batchsize_num/self.gpu_num
                    worker_num = int(worker.replace("worker", ""))
                    thread_num = float(thread.replace("thread", ""))
                    parse_filename_ori = f"{logtype}_{dataset}_{aug}_{model}_{epoch}_{batchsize}_{worker}_{thread}"
                    parse_filename = parse_filename_ori
                    df = pd.read_csv(
                        f"./{self.parsed_dir}/{parse_filename}.csv", index_col=None)

                    if 'Epoch' in df.columns:
                        df["Epoch"] = df["Epoch"].astype(int)

                    # split_df = split_df.iloc[1:]
                    # batch_df = batch_df.iloc[1:]

                    # print(parse_filename)

                    # Fliter out
                    # df = df[df["Epoch"] > 0]
                    # df = df[df["Epoch"] < (epoch_num-1)]

                    gpu_stall_col = []
                    stall_inspect_df = pd.DataFrame()
                    for i in range(self.gpu_num):
                        gpu_df = df[df["GPU"] == i]
                        # print(gpu_df)
                        gpu_stall_col.append(f"Training stall time (sec)_gpu{i}")
                        if 'Epoch' in df.columns:
                            # gpu_df = gpu_df[gpu_df["Epoch"]  0]
                            if i == 0:
                                stall_inspect_df = gpu_df[[
                                    "Epoch", "Step", "Training stall time (sec)"]]
                            else:
                                # print(gpu_df)
                                # print(fetch_inspect_df)
                                # print(start_inspect_df)
                                stall_inspect_df = stall_inspect_df.merge(right=gpu_df[["Epoch", "Step", "Training stall time (sec)"]], on=[
                                                                        "Epoch", "Step"], suffixes=('', f"_gpu{i}"))
                    if 'Epoch' in df.columns:
                        stall_inspect_df = stall_inspect_df.rename(
                            columns={'Training stall time (sec)': 'Training stall time (sec)_gpu0'})
                    
                        stall_inspect_df["Delay time diff (sec)"] = stall_inspect_df[gpu_stall_col].max(
                            axis=1) - stall_inspect_df[gpu_stall_col].min(axis=1)

                        stall_inspect_df.to_csv(
                            f"{self.parsed_dir}/{parse_filename}_stalldifftime.csv", sep=',', na_rep="nan")

                    # avg_epoch_time = round(df["Fetch time (sec)"].astype(
                    #     float).sum()/(int(epoch_num)-1), 2)
                    try:
                        avg_delayed_time = f'{round(stall_inspect_df["Delay time diff (sec)"].mean(),4)}±{round(stall_inspect_df["Delay time diff (sec)"].std(),4)}'
                    except:
                        avg_delayed_time = 'NA'
                        
                    # print(df.describe())
                    # print("\n")
                    # print(parse_filename, df["Iteration time (sec)"])

                    total_delay_loglet = [term[logtype], term[dataset], model, term[aug], worker_num, thread_num, epoch_num, batchsize_num,
                                          avg_delayed_time]
                    total_delay_log.append(total_delay_loglet)
            # print(total_log)
            # print(dataset_col_name)
            delay_avg_df = pd.DataFrame(total_delay_log,
                                        columns=delay_col_name)
            delay_avg_df.to_csv(self.parsed_dir+f"/stall_total_summary.csv",
                                sep=',', na_rep='NA')
        
        
class cpuParser(baseParser):
    def __init__(self):
        super(cpuParser, self).__init__("/cpu")
        breakdown_col_name = []
        for i in range(CPU_NUM):
            breakdown_col_name.append(f"CPU{i} Util (%)")

        breakdown_col_name.append(f"Socket0 Util (%)")
        breakdown_col_name.append(f"Socket1 Util (%)")
        breakdown_col_name.append(f"Total CPU Util (%)")
        self.breakdown_col_name = breakdown_col_name
        self.output_file = "pid.log"
        os.makedirs(self.parsed_dir, exist_ok=True)

    def parser(self):
        for tr_type in trainTypeSet:
            redefine(tr_type)
            self.set_dir()
            for logtype in logtypes:
                for logdir in self.dir_names[logtype]:
                    if not os.path.exists(logdir) or not os.path.isfile(logdir + "/" + self.output_file):
                        continue
                    logdir_list = logdir.split('/')

                    dataset = self.find_value(logdir_list, logtype)
                    aug = self.find_value(logdir_list, dataset)
                    model = self.find_value(logdir_list, aug)
                    epoch = self.find_value(logdir_list, model)
                    batchsize = self.find_value(logdir_list, epoch)
                    worker = self.find_value(logdir_list, batchsize)
                    thread = self.find_value(logdir_list, worker)
                    model = model.replace("_", "")

                    parse_filename = f"{logtype}_{dataset}_{aug}_{model}_{epoch}_{batchsize}_{worker}_{thread}_cpuutil"

                    logfile = logdir + "/" + self.output_file
                    breakdown_parsed_log = []
                    single_log = []
                    socket0_util_total = 0
                    socket1_util_total = 0
                    cpu_util_total = 0
                    for i in range(CPU_NUM):
                        single_log.append(0)

                    for line in open(logfile, 'r').readlines():
                        # start log
                        if line.find(f'python') != -1 or line.find(f'anaconda3/envs/') != -1:
                            replace_txt = line.replace('\n', '')
                            replace_txt = replace_txt.replace('[', '')
                            replace_txt = replace_txt.replace(']', '')
                            test = replace_txt.split(' ')
                            info = list(filter(lambda x: x != "", test))
                            info[-1] = "python"
                            try:
                                cpu_util = float(info[0])
                                cpu_util_total += cpu_util
                                cpu_num = int(info[5])
                                single_log[cpu_num] += cpu_util
                            except:
                                raise ValueError(f'{info}')
                                break
                            if cpu_num < CPU_NUM/2:
                                socket0_util_total += cpu_util
                            else:
                                socket1_util_total += cpu_util
                        elif line.find(f'Task') != -1:  # start log
                            single_log.extend(
                                [socket0_util_total, socket1_util_total, cpu_util_total])
                            breakdown_parsed_log.append(single_log)
                            single_log = []
                            for i in range(CPU_NUM):
                                single_log.append(0)
                            socket0_util_total = 0
                            socket1_util_total = 0
                            cpu_util_total = 0
                            # append
                        else:
                            pass

                    breakdown_df = pd.DataFrame(breakdown_parsed_log,
                                                columns=self.breakdown_col_name)
                    breakdown_df.dropna().to_csv(self.parsed_dir+"/"+parse_filename +
                                                 ".csv", sep=',', na_rep='NA')


class memParser(baseParser):
    def __init__(self):
        super(memParser, self).__init__("/mem")
        self.breakdown_col_name = ["Used memory (GB)", "Free memory (GB)", "Shared memory (GB)",
                                   "Buffer/Page cache (GB)", "Available memory (GB)", "Swap space used (GB)", "Swap space free (GB)"]
        self.output_file = "memory.log"
        os.makedirs(self.parsed_dir, exist_ok=True)

    def find_value(self, arr, target, jumpto=1):
        try:
            num = self.replace_str(arr[arr.index(target)+jumpto])
            if num.find("G") != -1:
                num = num.replace("G", "")
            elif num.find("M") != -1:
                num = num.replace("M", "")
                num = float(num) / 1024
        except:
            print(arr, target, arr[arr.index(target)+jumpto])
            num = 'NA'
        return num

    def parser(self):
        for tr_type in trainTypeSet:
            redefine(tr_type)
            self.set_dir()
            for logtype in logtypes:
                for logdir in self.dir_names[logtype]:
                    if not os.path.exists(logdir) or not os.path.isfile(logdir + "/" + self.output_file):
                        continue
                    logdir_list = logdir.split('/')

                    dataset = self.find_value(logdir_list, logtype)
                    aug = self.find_value(logdir_list, dataset)
                    model = self.find_value(logdir_list, aug)
                    epoch = self.find_value(logdir_list, model)
                    batchsize = self.find_value(logdir_list, epoch)
                    worker = self.find_value(logdir_list, batchsize)
                    thread = self.find_value(logdir_list, worker)
                    model = model.replace("_", "")

                    parse_filename = f"{logtype}_{dataset}_{aug}_{model}_{epoch}_{batchsize}_{worker}_{thread}_memory"

                    logfile = logdir + "/" + self.output_file
                    breakdown_parsed_log = []

                    mem_idx = 0
                    swap_idx = 0

                    for line in open(logfile, 'r').readlines():
                        if line.startswith(f'Mem:'):  # start log
                            replace_txt = line.replace('\t', ' ')
                            test = replace_txt.split(' ')
                            info = list(filter(lambda x: x != "", test))
                            used_memory = self.find_value(info, "Mem:", 2)
                            free_memory = self.find_value(info, "Mem:", 3)
                            shared_memory = self.find_value(info, "Mem:", 4)
                            buffer_memory = self.find_value(info, "Mem:", 5)
                            available_memory = self.find_value(info, "Mem:", 6)
                            breakdown_parsed_log.append(
                                [used_memory, free_memory, shared_memory, buffer_memory, available_memory])
                            mem_idx += 1
                        elif line.startswith(f'Swap:'):  # start log
                            replace_txt = line.replace('\t', ' ')
                            test = replace_txt.split(' ')
                            info = list(filter(lambda x: x != "", test))
                            used_memory = self.find_value(info, "Swap:", 2)
                            free_memory = self.find_value(info, "Swap:", 3)
                            breakdown_parsed_log[swap_idx].extend(
                                [used_memory, free_memory])
                            swap_idx += 1
                        else:
                            pass

                    breakdown_df = pd.DataFrame(breakdown_parsed_log,
                                                columns=self.breakdown_col_name)
                    breakdown_df.dropna().to_csv(self.parsed_dir+"/"+parse_filename +
                                                 ".csv", sep=',', na_rep='NA')


class ioParser(baseParser):
    def __init__(self, disk_name):
        super(ioParser, self).__init__("/io")
        self.breakdown_col_name = [
            "TPS", "KB read/s", "KB write/s", "KB read", "KB write"]
        self.output_file = "ssd_io.log"
        self.disk_name = disk_name
        os.makedirs(self.parsed_dir, exist_ok=True)

    def parser(self):
        for tr_type in trainTypeSet:
            redefine(tr_type)
            self.set_dir()
            for logtype in logtypes:
                for logdir in self.dir_names[logtype]:
                    if not os.path.exists(logdir) or not os.path.isfile(logdir + "/" + self.output_file):
                        continue
                    logdir_list = logdir.split('/')

                    dataset = self.find_value(logdir_list, logtype)
                    aug = self.find_value(logdir_list, dataset)
                    model = self.find_value(logdir_list, aug)
                    epoch = self.find_value(logdir_list, model)
                    batchsize = self.find_value(logdir_list, epoch)
                    worker = self.find_value(logdir_list, batchsize)
                    thread = self.find_value(logdir_list, worker)
                    model = model.replace("_", "")

                    parse_filename = f"{logtype}_{dataset}_{aug}_{model}_{epoch}_{batchsize}_{worker}_{thread}_io"

                    logfile = logdir + "/" + self.output_file
                    breakdown_parsed_log = []

                    for line in open(logfile, 'r').readlines():
                        if line.startswith(self.disk_name):  # start log
                            replace_txt = line.replace('\t', ' ')
                            test = replace_txt.split(' ')
                            info = list(filter(lambda x: x != "", test))
                            tps = self.find_value(info, self.disk_name)
                            kB_readPs = self.find_value(
                                info, self.disk_name, 2)
                            kB_wrtnPs = self.find_value(
                                info, self.disk_name, 3)
                            kB_read = self.find_value(info, self.disk_name, 4)
                            kB_wrtn = self.find_value(info, self.disk_name, 5)
                            breakdown_parsed_log.append(
                                [tps, kB_readPs, kB_wrtnPs, kB_read, kB_wrtn])
                        else:
                            pass

                    breakdown_df = pd.DataFrame(breakdown_parsed_log,
                                                columns=self.breakdown_col_name)
                    breakdown_df.dropna().to_csv(self.parsed_dir+"/"+parse_filename +
                                                 ".csv", sep=',', na_rep='NA')


class cacheParser(baseParser):
    def __init__(self):
        super(cacheParser, self).__init__("/cache")
        self.breakdown_col_name = [
            "Time", "Hits", "Misses", "Dirties", "Ratio", "Buffers (MB)", "Cache (MB)", "Hit ratio", "Miss ratio"]
        self.output_file = "cache.log"
        os.makedirs(self.parsed_dir, exist_ok=True)

    def parser(self):
        for tr_type in trainTypeSet:
            redefine(tr_type)
            self.set_dir()
            for logtype in logtypes:
                for logdir in self.dir_names[logtype]:
                    if not os.path.exists(logdir) or not os.path.isfile(logdir + "/" + self.output_file):
                        continue
                    logdir_list = logdir.split('/')

                    dataset = self.find_value(logdir_list, logtype)
                    aug = self.find_value(logdir_list, dataset)
                    model = self.find_value(logdir_list, aug)
                    epoch = self.find_value(logdir_list, model)
                    batchsize = self.find_value(logdir_list, epoch)
                    worker = self.find_value(logdir_list, batchsize)
                    thread = self.find_value(logdir_list, worker)
                    model = model.replace("_", "")

                    parse_filename = f"{logtype}_{dataset}_{aug}_{model}_{epoch}_{batchsize}_{worker}_{thread}_cache"

                    logfile = logdir + "/" + self.output_file
                    breakdown_parsed_log = []

                    for line in open(logfile, 'r').readlines():
                        if line.find("%") != -1:  # start log
                            replace_txt = line.replace('\t', ' ')
                            test = replace_txt.split(' ')
                            info = list(filter(lambda x: x != "", test))
                            time = info[0]
                            hits = info[1]
                            misses = info[2]
                            dirties = info[3]
                            ratio = info[4]
                            buffer = info[5]
                            cache = info[6].replace('\n', '')
                            miss_ratio = float(misses) / \
                                (float(hits) + float(misses))
                            hit_ratio = float(hits) / \
                                (float(hits) + float(misses))

                            breakdown_parsed_log.append(
                                [time, hits, misses, dirties, ratio, buffer, cache, hit_ratio, miss_ratio])
                        else:
                            pass

                    breakdown_df = pd.DataFrame(breakdown_parsed_log,
                                                columns=self.breakdown_col_name)
                    breakdown_df.dropna().to_csv(self.parsed_dir+"/"+parse_filename +
                                                 ".csv", sep=',', na_rep='NA')

    def total_summary(self):
        for tr_type in trainTypeSet:
            redefine(tr_type)
            self.set_dir()
            dataset_col_name = ["Log type", "Dataset size(GB)", "Augmentation", "Worker", "Epoch", "Batch size",
                                "Avg hit ratio", "Avg miss ratio", "Expected avg miss ratio"]

            total_log = []
            for logtype in logtypes:
                for logdir in self.dir_names[logtype]:
                    if not os.path.exists(logdir) or not os.path.isfile(logdir + "/" + self.output_file):
                        continue
                    logdir_list = logdir.split('/')
                    # print(logdir_list)
                    dataset = self.find_value(logdir_list, logtype)
                    aug = self.find_value(logdir_list, dataset)
                    model = self.find_value(logdir_list, aug)
                    epoch = self.find_value(logdir_list, model)
                    batchsize = self.find_value(logdir_list, epoch)
                    worker = self.find_value(logdir_list, batchsize)
                    thread = self.find_value(logdir_list, worker)
                    model = model.replace("_", "")
                    # print(batchsize)
                    epoch_num = int(epoch.replace("epoch", ""))
                    batchsize_num = int(batchsize.replace("b", ""))
                    single_batchsize_num = batchsize_num/self.gpu_num
                    worker_num = int(worker.replace("worker", ""))

                    parse_filename = f"{logtype}_{dataset}_{aug}_{model}_{epoch}_{batchsize}_{worker}_{thread}_cache"
                    df = pd.read_csv(
                        f"./{self.parsed_dir}/{parse_filename}.csv", index_col=None)
                    # print(parse_filename)

                    # Fliter out
                    df.drop(df.head(2000).index, inplace=True)
                    df.drop(df.tail(1000).index, inplace=True)

                    hitratio = df["Hit ratio"].mean()
                    missratio = df["Miss ratio"].mean()
                    try:
                        expected_miss = df["Cache (MB)"].mean(
                        )/1024.0/term[dataset]
                    except:
                        expected_miss = "nan"
                    total_loglet = [term[logtype], term[dataset], term[aug], worker_num, epoch_num, batchsize_num,
                                    hitratio, missratio, expected_miss]
                    total_log.append(total_loglet)

            avg_df = pd.DataFrame(total_log,
                                  columns=dataset_col_name)
            avg_df.dropna().to_csv(self.parsed_dir+f"/df_analyzer_total_summary.csv",
                                   sep=',', na_rep='NA')


class dsParser(baseParser):
    def __init__(self):
        super(dsParser, self).__init__("/ds")
        self.breakdown_col_name = [
            "Time", "Index queues", "Worker result queue", "Data queue", "Consumed"]
        self.output_file = "output.log"
        os.makedirs(self.parsed_dir, exist_ok=True)
        self.epoch = 5
        self.gpus = 4
        self.idx_count = self.epoch * self.gpus - self.gpus

    def parser(self):
        for tr_type in trainTypeSet:
            redefine(tr_type)
            self.set_dir()
            isPin = False
            index_queue_put_pattern = re.compile("PUT .* worker_queue_idx")
            index_queue_get_pattern = re.compile("worker_id: [0-9]+, GET")
            worker_result_queue_put_pattern = re.compile(
                "worker_id: [0-9]+, PUT")
            worker_result_queue_get_pattern = re.compile(
                "pin_memory GET .* _worker_result_queue")
            data_queue_put_pattern = re.compile("pin_memory PUT .* data_queue")
            data_queue_get_pattern = re.compile(
                "GET data object at .* _data_queue")
            consumption_pattern = re.compile("Finish iteration[0-9]+")
            datapoint_pattern = re.compile("Dataset GET_tar .* INDEX [0-9]+")
            fetch_start_pattern = re.compile("worker_id: [0-9]+, START FETCH")
            fetch_time_pattern = re.compile("worker_id: [0-9]+, END FETCH")

            for logtype in logtypes:
                if logtype.find("Persistent") != -1:
                    isPin = False
                else:
                    isPin = True
                for logdir in self.dir_names[logtype]:
                    if not os.path.exists(logdir) or not os.path.isfile(logdir + "/" + self.output_file):
                        continue
                    logdir_list = logdir.split('/')

                    dataset = self.find_value(logdir_list, logtype)
                    aug = self.find_value(logdir_list, dataset)
                    model = self.find_value(logdir_list, aug)
                    epoch = self.find_value(logdir_list, model)
                    batchsize = self.find_value(logdir_list, epoch)
                    worker = self.find_value(logdir_list, batchsize)
                    thread = self.find_value(logdir_list, worker)
                    model = model.replace("_", "")

                    worker_num = int(worker.replace("worker", ""))
                    parse_filename = f"{logtype}_{dataset}_{aug}_{model}_{epoch}_{batchsize}_{worker}_{thread}_ds"

                    logfile = logdir + "/" + self.output_file
                    breakdown_parsed_log = []
                    weird_idx_log = []
                    weird_idx_info = [0]
                    data_log = [0, 0, 0, 0, 0]
                    idx0_count = 0

                    worker_name = []
                    weird_idx_name = ["Time"]

                    # 2021-03-02 20:32:38.667 DEBUG:	Dataset GET_tar 1.2320932000875473e-05 GET_img 0.0005957740359008312 LOAD 0.009696600027382374 AUG 0.0021483160089701414 INDEX 6040
                    datapoint_time_name = [
                        "Data point num", "Get path (sec)", "Load img (sec)", "Decode (sec)", "Augmentation (sec)"]
                    datapoint_log = []

                    # 2021-03-02 20:32:38.556 DEBUG:	worker_id: 0, END FETCH at_time 1.912000646814704 data idx4
                    worker_fetch_name = ["Epoch", "Index number"]
                    worker_fetch_log = []
                    worker_fetch_loglet = [0, 0]
                    fetch_epoch_num = 0
                    jumper = 0

                    max_gpu_num = 4 if trainType.find("DDP") != -1 else 1
                    fetch_loglet = [-1, -1, -1, -
                                    1] if max_gpu_num == 4 else [-1]
                    for i in range(max_gpu_num):
                        worker_fetch_name.append(f"GPU{i} Fetch time (sec)")
                    for i in range(worker_num):
                        data_log.append(0)
                        worker_name.append(f"Worker{i} index queue")

                    for line in open(logfile, 'r').readlines():
                        if re.search(index_queue_put_pattern, line) is not None:  # start log
                            replace_txt = line.replace('\t', ' ')
                            test = replace_txt.split(' ')
                            info = list(filter(lambda x: x != "", test))

                            data_log[0] = self.find_value(
                                info, "DEBUG:", -1)  # Time
                            data_log[1] += 1
                            worker_queue = self.find_value(
                                info, "_index_queues:\n", -1)
                            worker_queue_num = int(
                                worker_queue.replace("worker_queue_idx", ""))
                            data_log[5+worker_queue_num] += 1

                            breakdown_parsed_log.append(data_log)
                            data_log = copy.deepcopy(data_log)

                            not_processed_batch_num = self.find_value(
                                info, "PUT")
                            not_processed_batch_num = not_processed_batch_num.replace(
                                "(", "")
                            batch_num = int(
                                not_processed_batch_num.replace(",", ""))
                            if batch_num >= len(weird_idx_info)-1:
                                weird_idx_name.append(f"Batch Idx {batch_num}")
                                weird_idx_info.append(0)

                        elif re.search(index_queue_get_pattern, line) is not None:  # start log
                            replace_txt = line.replace('\t', ' ')
                            test = replace_txt.split(' ')
                            info = list(filter(lambda x: x != "", test))

                            data_log[0] = self.find_value(
                                info, "DEBUG:", -1)  # Time
                            data_log[1] -= 1
                            worker_queue = self.find_value(info, "worker_id:")
                            worker_queue_num = int(
                                worker_queue.replace(",", ""))
                            data_log[5+worker_queue_num] -= 1

                            breakdown_parsed_log.append(data_log)
                            data_log = copy.deepcopy(data_log)
                        elif re.search(worker_result_queue_put_pattern, line) is not None:  # start log
                            replace_txt = line.replace('\t', ' ')
                            test = replace_txt.split(' ')
                            info = list(filter(lambda x: x != "", test))

                            data_log[0] = self.find_value(
                                info, "DEBUG:", -1)  # Time
                            data_log[2] += 1

                            breakdown_parsed_log.append(data_log)
                            data_log = copy.deepcopy(data_log)
                        elif re.search(worker_result_queue_get_pattern, line) is not None:  # start log
                            replace_txt = line.replace('\t', ' ')
                            test = replace_txt.split(' ')
                            info = list(filter(lambda x: x != "", test))

                            data_log[0] = self.find_value(
                                info, "DEBUG:", -1)  # Time
                            data_log[2] -= 1

                            breakdown_parsed_log.append(data_log)
                            data_log = copy.deepcopy(data_log)

                        elif re.search(data_queue_put_pattern, line) is not None:  # start log
                            replace_txt = line.replace('\t', ' ')
                            test = replace_txt.split(' ')
                            info = list(filter(lambda x: x != "", test))

                            data_log[0] = self.find_value(
                                info, "DEBUG:", -1)  # Time
                            weird_idx_info[0] = data_log[0]
                            data_log[3] += 1

                            breakdown_parsed_log.append(data_log)
                            data_log = copy.deepcopy(data_log)

                            not_processed_batch_num = self.find_value(
                                info, "PUT")
                            not_processed_batch_num = not_processed_batch_num.replace(
                                "(", "")
                            batch_num = int(
                                not_processed_batch_num.replace(",", ""))
                            try:
                                weird_idx_info[1+batch_num] += 1
                            except:
                                weird_idx_info.insert(1+batch_num, 1)
                                # print(f"{parse_filename}: pin_memory PUT .* data_queue pattern error")
                                # print(batch_num)
                                # print(weird_idx_info)
                                # print(weird_idx_log)
                                # return

                            weird_idx_log.append(weird_idx_info)
                            weird_idx_info = copy.deepcopy(weird_idx_info)
                        elif re.search(data_queue_get_pattern, line) is not None:  # start log
                            replace_txt = line.replace('\t', ' ')
                            test = replace_txt.split(' ')
                            info = list(filter(lambda x: x != "", test))

                            data_log[0] = self.find_value(
                                info, "DEBUG:", -1)  # Time
                            data_log[3] -= 1

                            breakdown_parsed_log.append(data_log)
                            data_log = copy.deepcopy(data_log)
                        elif re.search(consumption_pattern, line) is not None:  # start log
                            replace_txt = line.replace('\t', ' ')
                            test = replace_txt.split(' ')
                            info = list(filter(lambda x: x != "", test))

                            data_log[0] = self.find_value(
                                info, "DEBUG:", -1)  # Time
                            data_log[4] += 1

                            breakdown_parsed_log.append(data_log)
                            data_log = copy.deepcopy(data_log)

                        elif re.search(datapoint_pattern, line) is not None:  # start log
                            # if idx0_count < self.idx_count:
                            #     continue
                            datapoint_time_loglet = [0, 0, 0, 0, 0]
                            replace_txt = line.replace('\t', ' ')
                            test = replace_txt.split(' ')
                            info = list(filter(lambda x: x != "", test))

                            datapoint_time_loglet[0] = self.find_value(
                                info, "INDEX")  # INDEX
                            datapoint_time_loglet[1] = self.find_value(
                                info, "GET_tar")  # GET target path
                            datapoint_time_loglet[2] = self.find_value(
                                info, "GET_img")  # GET img in mem
                            datapoint_time_loglet[3] = self.find_value(
                                info, "LOAD")  # Load and decode
                            datapoint_time_loglet[4] = self.find_value(
                                info, "AUG")  # Augmentation

                            datapoint_log.append(datapoint_time_loglet)

                        elif re.search(fetch_start_pattern, line) is not None:  # start log
                            replace_txt = line.replace('\t', ' ')
                            test = replace_txt.split(' ')
                            info = list(filter(lambda x: x != "", test))
                            # if self.find_value(info, "data").replace("idx","") == "0":
                            #     idx0_count += 1
                        elif re.search(fetch_time_pattern, line) is not None:  # start log
                            replace_txt = line.replace('\t', ' ')
                            test = replace_txt.split(' ')
                            info = list(filter(lambda x: x != "", test))

                            # worker_fetch_loglet[0] = self.find_value(
                            #     info, "worker_id:")  # Worker id
                            idx_value = int(self.find_value(
                                info, "data").replace("idx", ""))  # Data idx

                            try:
                                gpu_num = int(self.find_value(
                                    info, "GPU:"))  # GPU num
                            except:
                                gpu_num = 0

                            fetch_time = self.find_value(
                                info, "at_time")  # Fetch time

                            if worker_fetch_loglet[1] == idx_value:
                                fetch_loglet[gpu_num] = fetch_time
                            else:
                                if jumper+idx_value < len(worker_fetch_log):
                                    try:
                                        assert worker_fetch_log[jumper +
                                                                idx_value][1] == idx_value
                                    except:
                                        print(
                                            worker_fetch_log[jumper+idx_value][1])
                                        print(len(worker_fetch_log))
                                        print(jumper)
                                        print(idx_value)
                                        raise EOFError
                                    worker_fetch_log[jumper +
                                                     idx_value][2+gpu_num] = fetch_time
                                else:
                                    print(idx_value)
                                    worker_fetch_loglet.extend(fetch_loglet)
                                    worker_fetch_log.append(
                                        worker_fetch_loglet)
                                    worker_fetch_loglet = [
                                        fetch_epoch_num, idx_value]
                                    fetch_loglet = [-1, -1, -1, -
                                                    1] if max_gpu_num == 4 else [-1]
                                    fetch_loglet[gpu_num] = fetch_time
                                    if idx_value == 0:
                                        fetch_epoch_num += 1
                                        worker_fetch_loglet[0] = fetch_epoch_num
                                        jumper = len(worker_fetch_log)
                        else:
                            pass
                        # at_time
                    current_col_name = self.breakdown_col_name + worker_name
                    breakdown_df = pd.DataFrame(breakdown_parsed_log,
                                                columns=current_col_name)

                    if isPin is False:
                        breakdown_df["Worker result queue"] = breakdown_df["Worker result queue"] - \
                            breakdown_df["Consumed"]
                        breakdown_df["Data queue"] = breakdown_df["Worker result queue"]

                    breakdown_df.to_csv(
                        self.parsed_dir+"/"+parse_filename+".csv", sep=',', na_rep='NA')

                    weird_idx_df = pd.DataFrame(weird_idx_log,
                                                columns=weird_idx_name)

                    # if isPin is False:
                    #     weird_idx_df["Worker result queue"] = weird_idx_df["Worker result queue"] - weird_idx_df["Consumed"]
                    #     weird_idx_df["Data queue"] = weird_idx_df["Worker result queue"]

                    weird_idx_df.to_csv(
                        self.parsed_dir+"/"+parse_filename+"WeirdIdx.csv", sep=',', na_rep=0)

                    # if idx0_count >= self.idx_count:
                    datapoint_df = pd.DataFrame(datapoint_log,
                                                columns=datapoint_time_name)
                    datapoint_df.to_csv(
                        self.parsed_dir+"/"+parse_filename+"Datapoint.csv", sep=',', na_rep=0)

                    worker_fetch_df = pd.DataFrame(worker_fetch_log,
                                                   columns=worker_fetch_name)
                    # worker_fetch_only_df = worker_fetch_df[["GPU1 Fetch time (sec)", "GPU2 Fetch time (sec)", "GPU3 Fetch time (sec)", "GPU4 Fetch time (sec)"]]
                    # worker_fetch_df["Max - Min (ms)"] = (worker_fetch_only_df.idxmax() - worker_fetch_only_df.idxmin()) * 1000
                    worker_fetch_df.to_csv(
                        self.parsed_dir+"/"+parse_filename+"fetch.csv", sep=',', na_rep=-1)


class simpleParser(baseParser):
    def __init__(self, output_file="output.log"):
        super(simpleParser, self).__init__("/simp")
        self.output_file = output_file
        os.makedirs(self.parsed_dir, exist_ok=True)

    def parser(self):
        fetch_time_pattern = re.compile("worker_id: [0-9]+, END FETCH at_time")
        split_time_pattern = re.compile("GPU: [0-9]+ Dataset GET_tar")
        decode_load_time_pattern = re.compile("Dataset FETCH")
        aggr_time_pattern = re.compile("GPU: [0-9]+ Processing data:")

        batch_time_pattern = re.compile("INFO:		Collate END at_time")
        for tr_type in trainTypeSet:
            redefine(tr_type)
            self.set_dir()

            for logtype in logtypes:
                for logdir in self.dir_names[logtype]:
                    if not os.path.exists(logdir) or not os.path.isfile(logdir + "/" + self.output_file):
                        setter = os.path.isfile(
                            logdir + "/" + self.output_file)
                        print(
                            f"os.path.exists: {os.path.exists(logdir)}, os.path.isfile: {setter}")
                        continue
                    logdir_list = logdir.split('/')

                    dataset = self.find_value(logdir_list, logtype)
                    aug = self.find_value(logdir_list, dataset)
                    model = self.find_value(logdir_list, aug)
                    epoch = self.find_value(logdir_list, model)
                    batchsize = self.find_value(logdir_list, epoch)
                    worker = self.find_value(logdir_list, batchsize)
                    thread = self.find_value(logdir_list, worker)
                    model = model.replace("_", "")

                    parse_filename = f"{logtype}_{dataset}_{aug}_{model}_{epoch}_{batchsize}_{worker}_{thread}_simp"

                    logfile = logdir + "/" + self.output_file
                    worker_fetch_name = ["Start time", "Time", "GPU", "Worker",
                                         "Index number", "Fetch time (sec)"]
                    worker_fetch_log = []

                    aggr_name = ["Time", "GPU", "Worker",
                                         "Index number", "Aggr time (sec)"]
                    aggr_log = []

                    split_name = [
                        "GPU", "Image number", "Fetch (sec)", "Decode (sec)", "Get target (sec)", "Get image (sec)", "Load (sec)", "Augmentation (sec)"]
                    split_log = []

                    batching_name = ["Batch time (sec)"]
                    batching_log = []

                    for line in open(logfile, 'r').readlines():
                        # print(line)
                        if re.search(fetch_time_pattern, line) is not None:  # start log
                            replace_txt = line.replace('\t', ' ')
                            test = replace_txt.split(' ')
                            info = list(filter(lambda x: x != "", test))
                            # if self.find_value(info, "data").replace("idx","") == "0":
                            #     idx0_count += 1
                            # worker_fetch_loglet[0] = self.find_value(
                            #     info, "worker_id:")  # Worker id
                            time_str1 = self.find_value(info, "DEBUG:", -2)
                            time_str2 = self.find_value(info, "DEBUG:", -1)
                            time_str = f"{time_str1} {time_str2}".split(
                                ")")[-1]

                            time = datetime.strptime(
                                time_str, '%Y-%m-%d %H:%M:%S.%f')
                            idx_value = int(self.find_value(
                                info, "data").replace("idx", ""))  # Data idx

                            try:
                                gpu_num = int(self.find_value(
                                    info, "GPU:"))  # GPU num
                            except:
                                gpu_num = 0
                            worker_num = int(
                                self.find_value(info, "worker_id:"))
                            fetch_time = self.find_value(
                                info, "at_time")  # Fetch time
                            previous_time = time - \
                                timedelta(seconds=float(fetch_time))
                            worker_fetch_log.append(
                                [previous_time, time, gpu_num, worker_num, idx_value, fetch_time])
                        elif re.search(split_time_pattern, line) is not None:  # start log
                            replace_txt = line.replace('\t', ' ')
                            test = replace_txt.split(' ')
                            info = list(filter(lambda x: x != "", test))

                            get_tar = float(self.find_value(
                                info, "GET_tar"))  # Data idx

                            get_img = float(self.find_value(
                                info, "GET_img"))  # Data idx

                            fetch = float(self.find_value(
                                info, "FETCH"))  # Data idx
                            
                            decode = float(self.find_value(
                                info, "DECODE"))  # Data idx
                            
                            load = float(self.find_value(
                                info, "LOAD"))  # Data idx

                            aug = float(self.find_value(
                                info, "AUG"))  # Data idx

                            idx_value = int(self.find_value(
                                info, "INDEX"))  # Data idx

                            try:
                                gpu_num = int(self.find_value(
                                    info, "GPU:"))  # GPU num
                            except:
                                gpu_num = 0

                            split_log.append(
                                [gpu_num, idx_value, fetch, decode, get_tar, get_img, load, aug])
                        elif re.search(batch_time_pattern, line) is not None:  # start log
                            replace_txt = line.replace('\t', ' ')
                            test = replace_txt.split(' ')
                            info = list(filter(lambda x: x != "", test))

                            batch_time = float(self.find_value(
                                info, "at_time"))  # Data idx

                            batching_log.append([batch_time])
                        elif re.search(aggr_time_pattern, line) is not None:  # start log
                            replace_txt = line.replace('\t', ' ')
                            test = replace_txt.split(' ')
                            info = list(filter(lambda x: x != "", test))

                            time_str1 = self.find_value(info, "DEBUG:", -2)
                            time_str2 = self.find_value(info, "DEBUG:", -1)
                            time_str = f"{time_str1} {time_str2}".split(
                                ")")[-1]

                            time = datetime.strptime(
                                time_str, '%Y-%m-%d %H:%M:%S.%f')

                            idx_value = int(self.find_value(
                                info, "idx"))

                            try:
                                gpu_num = int(self.find_value(
                                    info, "GPU:"))  # GPU num
                            except:
                                gpu_num = 0
                            # worker_num = int(
                            #     self.find_value(info, "worker_id:"))

                            aggr_time = self.find_value(
                                info, "data:")  # Fetch time
                            aggr_log.append(
                                [time, gpu_num, "NA", idx_value, aggr_time])
                    worker_fetch_df = pd.DataFrame(worker_fetch_log,
                                                   columns=worker_fetch_name)
                    new_worker_fetch_df = pd.DataFrame()
                    
                    max_iter = worker_fetch_df["Index number"].max()
                    
                    for i in range(self.gpu_num):
                        iter_count = 0
                        epoch_count = 0
                        gpu_df = worker_fetch_df[worker_fetch_df["GPU"] == i]
                        
                        for idx, row in gpu_df.iterrows():
                            gpu_df.loc[idx,'Epoch'] = epoch_count
                            iter_count += 1
                            if iter_count > max_iter:
                                iter_count = 0
                                epoch_count += 1

                        new_worker_fetch_df = pd.concat([new_worker_fetch_df,gpu_df])

                    # worker_fetch_df["Max - Min (ms)"] = (worker_fetch_only_df.idxmax() - worker_fetch_only_df.idxmin()) * 1000
                    new_worker_fetch_df.to_csv(
                        self.parsed_dir+"/"+parse_filename+".csv", sep=',', na_rep=-1)

                    aggr_time_df = pd.DataFrame(aggr_log,
                                                columns=aggr_name)
                    aggr_time_df.to_csv(
                        self.parsed_dir+"/"+parse_filename+"_aggr.csv", sep=',', na_rep=-1)

                    split_df = pd.DataFrame(split_log,
                                            columns=split_name)
                    split_df.to_csv(
                        self.parsed_dir+"/"+parse_filename+"_split.csv", sep=',', na_rep=-1)

                    batch_df = pd.DataFrame(batching_log,
                                            columns=batching_name)
                    batch_df.to_csv(
                        self.parsed_dir+"/"+parse_filename+"_batch.csv", sep=',', na_rep=-1)

    def total_summary(self):
        for tr_type in trainTypeSet:
            redefine(tr_type)
            self.set_dir()
            dataset_col_name = ["Log type", "Dataset size(GB)", "Model", "Augmentation", "Worker", "Worker batch size", "Epoch", "Batch size",
                                "Avg fetch time (sec)", "Avg batch fecth time (sec)", "Avg aggregation time (ms)", "Batch aggregation time (ms)"]
            delay_col_name = ["Log type", "Dataset size(GB)", "Model", "Augmentation", "Worker", "Worker batch size", "Epoch", "Batch size",
                              "Avg delayed time (sec)", "Avg fetch diff (sec)", "Avg start fetch diff (sec)"]
            dataset_gpu_name = [
                "Log type", "Dataset size(GB)", "Model", "Augmentation", "Worker", "Worker batch size", "Epoch", "Batch size"]

            for i in range(self.gpu_num):
                dataset_col_name.append(f"GPU{i} Avg fetch time (sec)")
                dataset_gpu_name.append(
                    f"GPU{i} batchSize * Avg Get target (sec)")
                dataset_gpu_name.append(
                    f"GPU{i} batchSize * Avg Get image (sec)")
                dataset_gpu_name.append(
                    f"GPU{i} batchSize * Avg Fetch (sec)")
                dataset_gpu_name.append(
                    f"GPU{i} batchSize * Avg Decode (sec)")
                dataset_gpu_name.append(f"GPU{i} batchSize * Avg Load (sec)")
                dataset_gpu_name.append(
                    f"GPU{i} batchSize * Avg Read and Decode (sec)")
                dataset_gpu_name.append(
                    f"GPU{i} batchSize * Avg Augmentation (sec)")
                dataset_gpu_name.append(f"GPU{i} batchSize * Avg Batch (sec)")

            total_log = []
            total_split_log = []
            total_delay_log = []
            for logtype in logtypes:
                for logdir in self.dir_names[logtype]:
                    if not os.path.exists(logdir) or not os.path.isfile(logdir + "/" + self.output_file):
                        continue
                    logdir_list = logdir.split('/')
                    # print(logdir_list)
                    dataset = self.find_value(logdir_list, logtype)
                    aug = self.find_value(logdir_list, dataset)
                    model = self.find_value(logdir_list, aug)
                    epoch = self.find_value(logdir_list, model)
                    batchsize = self.find_value(logdir_list, epoch)
                    worker = self.find_value(logdir_list, batchsize)
                    thread = self.find_value(logdir_list, worker)
                    model = model.replace("_", "")
                    # print(batchsize)
                    epoch_num = int(epoch.replace("epoch", ""))
                    batchsize_num = int(batchsize.replace("b", ""))
                    single_batchsize_num = batchsize_num/self.gpu_num
                    worker_num = int(worker.replace("worker", ""))
                    thread_num = float(thread.replace("thread", ""))
                    parse_filename_ori = f"{logtype}_{dataset}_{aug}_{model}_{epoch}_{batchsize}_{worker}_{thread}"
                    parse_filename = f"{parse_filename_ori}_simp"
                    df = pd.read_csv(
                        f"./{self.parsed_dir}/{parse_filename}.csv", index_col=None)
                    df["Start time"] = pd.to_datetime(
                                            df["Start time"], format='%Y-%m-%d %H:%M:%S.%f', errors='ignore')
                    if 'Epoch' in df.columns:
                        df["Epoch"] = df["Epoch"].astype(int)
                    aggr_df = pd.read_csv(
                        f"./{self.parsed_dir}/{parse_filename}_aggr.csv", index_col=None)

                    split_df = pd.read_csv(
                        f"./{self.parsed_dir}/{parse_filename}_split.csv", index_col=None)

                    batch_df = pd.read_csv(
                        f"./{self.parsed_dir}/{parse_filename}_batch.csv", index_col=None)

                    try:
                        delay_df = pd.read_csv(
                            f"./dsanalyzer_parsed/{suffix_logdir}/{trainType}/{parse_filename_ori}_delayedtime.csv", index_col=None)
                    except:
                        delay_df = pd.DataFrame()
                    # split_df = split_df.iloc[1:]
                    # batch_df = batch_df.iloc[1:]

                    # print(parse_filename)

                    # Fliter out
                    # df = df[df["Epoch"] > 0]
                    # df = df[df["Epoch"] < (epoch_num-1)]

                    gpu_log = []
                    gpu_log_split = []
                    gpu_start_col = []
                    gpu_fetch_col = []
                    fetch_inspect_df = pd.DataFrame()
                    start_inspect_df = pd.DataFrame()
                   
                    for i in range(self.gpu_num):
                        gpu_df = df[df["GPU"] == i]
                        
                        # print(gpu_df)
                        gpu_epoch_time = f'{round(gpu_df["Fetch time (sec)"].mean(),4)}±{round(gpu_df["Fetch time (sec)"].std(),4)}'

                        
                        gpu_fetch_col.append(f"Fetch time (sec)_gpu{i}")
                        gpu_start_col.append(f"Start time_gpu{i}")
                        
                        gpu_log.append(gpu_epoch_time)
                        
                        if 'Epoch' in df.columns:
                            gpu_df = gpu_df[gpu_df["Epoch"] > 0]
                            if i == 0:
                                fetch_inspect_df = gpu_df[[
                                    "Epoch", "Index number", "Fetch time (sec)"]]
                                start_inspect_df = gpu_df[[
                                    "Epoch", "Index number", "Start time"]]
                            else:
                                # print(gpu_df)
                                # print(fetch_inspect_df)
                                # print(start_inspect_df)
                                fetch_inspect_df = fetch_inspect_df.merge(right=gpu_df[["Epoch", "Index number", "Fetch time (sec)"]], on=[
                                                                        "Epoch", "Index number"], suffixes=('', f"_gpu{i}"))
                                start_inspect_df = start_inspect_df.merge(right=gpu_df[["Epoch", "Index number", "Start time"]], on=[
                                                                        "Epoch", "Index number"], suffixes=('', f"_gpu{i}"))
                        gpu_split_df = split_df[split_df["GPU"] == i].dropna()

                        get_target = gpu_split_df["Get target (sec)"].mean(
                        )*single_batchsize_num
                        get_image = gpu_split_df["Get image (sec)"].mean(
                        )*single_batchsize_num
                        get_load = gpu_split_df["Load (sec)"].mean(
                        )*single_batchsize_num
                        read_n_decode = get_target + get_image + get_load
                        get_augment = gpu_split_df["Augmentation (sec)"].mean(
                        )*single_batchsize_num
                        get_fetch = gpu_split_df["Fetch (sec)"].mean(
                        )*single_batchsize_num
                        get_decode = gpu_split_df["Decode (sec)"].mean(
                        )*single_batchsize_num
                        # print(gpu_split_df.describe())
                        batching = batch_df["Batch time (sec)"].mean()
                        
                        gpu_log_split.extend([
                            get_target,
                            get_image,
                            get_fetch,
                            get_decode,
                            get_load,
                            read_n_decode,
                            get_augment,
                            batching
                        ])
                        
                    
                    if 'Epoch' in df.columns:
                        fetch_inspect_df = fetch_inspect_df.rename(
                            columns={'Fetch time (sec)': 'Fetch time (sec)_gpu0'})
                        start_inspect_df = start_inspect_df.rename(
                            columns={'Start time': 'Start time_gpu0'})
                    
                        fetch_inspect_df["Batch fetch time diff (sec)"] = fetch_inspect_df[gpu_fetch_col].max(
                            axis=1) - fetch_inspect_df[gpu_fetch_col].min(axis=1)

                        start_inspect_df["Batch fetch start time diff (sec)"] = start_inspect_df[gpu_start_col].max(
                            axis=1) - start_inspect_df[gpu_start_col].min(axis=1)
                        start_inspect_df["Batch fetch start time diff (sec)"] = start_inspect_df["Batch fetch start time diff (sec)"].dt.total_seconds(
                        )
                        fetch_inspect_df.to_csv(
                            f"{self.parsed_dir}/{parse_filename}_fetchdifftime.csv", sep=',', na_rep="nan")
                        start_inspect_df.to_csv(
                            f"{self.parsed_dir}/{parse_filename}_fetchstartdifftime.csv", sep=',', na_rep="nan")

                    # avg_epoch_time = round(df["Fetch time (sec)"].astype(
                    #     float).sum()/(int(epoch_num)-1), 2)
                    try:
                        avg_delayed_time = f'{round(delay_df["Delayed time"].mean(),4)}±{round(delay_df["Delayed time"].std(),4)}'
                    except:
                        avg_delayed_time = 'NA'
                    # print(df.describe())
                    # print("\n")
                    # print(parse_filename, df["Iteration time (sec)"])
                    need_worker_batch = 1
                    if thread_num != 0:
                        need_worker_batch = (
                            single_batchsize_num // thread_num)
                    if not fetch_inspect_df.empty:
                        fetch_diff_avg = f'{round(fetch_inspect_df["Batch fetch time diff (sec)"].mean(),4)}±{round(fetch_inspect_df["Batch fetch time diff (sec)"].std(),4)}'
                    else:
                        fetch_diff_avg = "NA"
                    if not start_inspect_df.empty:
                        fetch_start_diff_avg = f'{round(start_inspect_df["Batch fetch start time diff (sec)"].mean(),4)}±{round(start_inspect_df["Batch fetch start time diff (sec)"].std(),4)}'
                    else:
                        fetch_start_diff_avg = "NA"
                    fetch_avg = f'{round(df["Fetch time (sec)"].mean(),4)}±{round(df["Fetch time (sec)"].std(),4)}'
                    batch_fetch_avg = f'{round(df["Fetch time (sec)"].mean() * need_worker_batch, 4)}'
                    aggr_avg = f'{round(aggr_df["Aggr time (sec)"].mean()*1000,2)}±{round(aggr_df["Aggr time (sec)"].std()*1000,2)}'
                    aggr_total = f'{round(aggr_df["Aggr time (sec)"].mean() * need_worker_batch * 1000, 4)}'
                    # # [Note] :
                    # # Deprecated throughput avg,
                    # # theoretically wrong in mathematic, Please check below link
                    # # https://fxloader.com/inverse_of_an_average_compared_to_averages_of_inverses/
                    # throughput_avg = f'{round(df["Throughput (image/sec)"].mean(),2)}±{round(df["Throughput (image/sec)"].std(),4)}'

                    total_loglet = [term[logtype], term[dataset], model, term[aug], worker_num, thread_num, epoch_num, batchsize_num,
                                    fetch_avg, batch_fetch_avg, aggr_avg, aggr_total]
                    total_loglet.extend(gpu_log)
                    total_log.append(total_loglet)

                    total_split_loglet = [
                        term[logtype], term[dataset], model, term[aug], worker_num, thread_num, epoch_num, batchsize_num]
                    total_delay_loglet = [term[logtype], term[dataset], model, term[aug], worker_num, thread_num, epoch_num, batchsize_num,
                                          avg_delayed_time, fetch_diff_avg, fetch_start_diff_avg]
                    total_delay_log.append(total_delay_loglet)
                    total_split_loglet.extend(gpu_log_split)
                    total_split_log.append(total_split_loglet)
            # print(total_log)
            # print(dataset_col_name)
            avg_df = pd.DataFrame(total_log,
                                  columns=dataset_col_name)
            avg_df.dropna().to_csv(self.parsed_dir+f"/worker_total_summary.csv",
                                   sep=',', na_rep='NA')

            split_df = pd.DataFrame(total_split_log,
                                    columns=dataset_gpu_name)
            split_df.to_csv(self.parsed_dir+f"/split_total_summary.csv",
                            sep=',', na_rep='NA')
            delay_avg_df = pd.DataFrame(total_delay_log,
                                        columns=delay_col_name)
            delay_avg_df.to_csv(self.parsed_dir+f"/delayed_total_summary.csv",
                                sep=',', na_rep='NA')

    def lifetime_parser(self):
        for tr_type in trainTypeSet:
            redefine(tr_type)
            self.set_dir()

            for logtype in logtypes:
                for logdir in self.dir_names[logtype]:
                    if not os.path.exists(logdir) or not os.path.isfile(logdir + "/" + self.output_file):
                        setter = os.path.isfile(
                            logdir + "/" + self.output_file)
                        print(
                            f"os.path.exists: {os.path.exists(logdir)}, os.path.isfile: {setter}")
                        continue
                    logdir_list = logdir.split('/')

                    dataset = self.find_value(logdir_list, logtype)
                    aug = self.find_value(logdir_list, dataset)
                    model = self.find_value(logdir_list, aug)
                    epoch = self.find_value(logdir_list, model)
                    batchsize = self.find_value(logdir_list, epoch)
                    worker = self.find_value(logdir_list, batchsize)
                    thread = self.find_value(logdir_list, worker)
                    model = model.replace("_", "")

                    worker_num = int(worker.replace("worker", ""))
                    parse_filename = f"{logtype}_{dataset}_{aug}_{model}_{epoch}_{batchsize}_{worker}_{thread}"

                    logfile = logdir + "/" + self.output_file

                    perf_data = pd.read_csv(f"./dsanalyzer_parsed/{suffix_logdir}/{trainType}/"+parse_filename+".csv",
                                            sep=',').dropna()
                    if perf_data.empty:
                        continue
                    simp_data = pd.read_csv(f"./dsanalyzer_parsed/{suffix_logdir}/{trainType}/simp/"+parse_filename+"_simp.csv",
                                            sep=',')
                    perf_data["Index number"] = perf_data["Step"]
                    perf_data = perf_data[[
                        "Time", "GPU", "Epoch", "Index number"]]
                    simp_data = simp_data[["Time", "GPU", "Index number"]]
                    merged_data = pd.merge(perf_data, simp_data, how="inner", on=[
                                           "GPU", "Index number"], suffixes=('_perf', "_simp_data"))
                    # print(merged_data)

                    merged_data['Time_perf'] = pd.to_datetime(
                        merged_data['Time_perf'], format='%Y-%m-%d %H:%M:%S.%f', errors='raise')
                    merged_data['Time_simp_data'] = pd.to_datetime(
                        merged_data['Time_simp_data'], format='%Y-%m-%d %H:%M:%S.%f', errors='raise')

                    merged_data["Batch life time (sec)"] = (
                        merged_data["Time_perf"] - merged_data["Time_simp_data"])

                    merged_data["Batch life time (sec)"] = merged_data["Batch life time (sec)"].dt.total_seconds(
                    )
                    # merged_data = merged_data[merged_data["Batch life time (sec)"]
                    #                       < 40.0]
                    merged_data.to_csv(
                        f"./dsanalyzer_parsed/{suffix_logdir}/{trainType}/{parse_filename}_batchlifetime.csv", sep=',', na_rep="nan")

                    merged_data = merged_data[[
                        "Epoch", "Index number", "GPU", "Time_perf"]]
                    perf_data['Time'] = pd.to_datetime(
                        perf_data['Time'], format='%Y-%m-%d %H:%M:%S.%f', errors='raise')
                    delayed_df = pd.DataFrame()
                    # print(parse_filename)
                    # print(perf_data)
                    gpu_col = []
                    for i in range(self.gpu_num):
                        gpu_col.append(f"Time_gpu{i}")
                        gpu_df = perf_data[perf_data["GPU"] == i]
                        gpu_df = gpu_df.sort_values(
                            by=["Epoch", "Index number"]).copy()
                        # print(gpu_df)
                        if i == 0:
                            delayed_df = gpu_df[[
                                "Epoch", "Index number", "Time"]]
                        else:
                            delayed_df = delayed_df.merge(right=gpu_df[["Epoch", "Index number", "Time"]], on=[
                                                          "Epoch", "Index number"], suffixes=('', f"_gpu{i}"))

                        # print(delayed_df)

                    delayed_df = delayed_df.rename(
                        columns={'Time': 'Time_gpu0'})

                    delayed_df["Delayed time"] = delayed_df[gpu_col].max(
                        axis=1) - delayed_df[gpu_col].min(axis=1)
                    delayed_df["Delayed time"] = delayed_df["Delayed time"].dt.total_seconds(
                    )

                    delayed_df.to_csv(
                        f"./dsanalyzer_parsed/{suffix_logdir}/{trainType}/{parse_filename}_delayedtime.csv", sep=',', na_rep="nan")
                                                                                                                                            