import matplotlib.pyplot as plt
import pandas as pd

from cycler import cycler
import os
import glob
from . import parsePackage
from multiprocessing import Pool
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)

plt.rc('font', family="DejaVu Serif", serif='Times')
plt.rc('text', usetex=False)
plt.rc('xtick', labelsize=28)
plt.rc('ytick', labelsize=28)
plt.rc('axes', labelsize=32)

small = (4, 4)
medium = (6, 6)
large = (12, 4)
xlarge = (12, 6)

term = parsePackage.figterm
trainType = parsePackage.trainType
parsed_dir = parsePackage.parsed_dir
suffix_logdir = parsePackage.suffix_logdir

savefolder = f"./fig_update/{suffix_logdir}/"
os.makedirs(savefolder, exist_ok=True)

def statistics(data):
    log.debug("max: ", data.max())
    log.debug("min: ",data.min())
    log.debug("mean: ",data.mean())
    log.debug("median: ",data.median())
    log.debug("std: ",data.std())
    log.debug("10large: ",data.nlargest(10))
    log.debug("10small: ",data.nsmallest(10))
    log.debug("quantile",data.quantile([0,.25, .5, .75,1.0]))

def set_dir(tr_type):
    global trainType
    global parsed_dir
    
    trainType = tr_type
    parsed_dir = f"dsanalyzer_parsed/{suffix_logdir}/{trainType}"    

class basePlotter():
    def __init__(self, figsize=xlarge):
        ###############
        # Figure
        ###############
        plt.rcParams["figure.figsize"] = figsize
        plt.rc('xtick', labelsize=32)
        plt.rc('ytick', labelsize=32)
        plt.rc('axes', labelsize=36)
        self.titlesize = 40
        self.linewidth = 1.5
        self.data = None
        self.legendfontsize = 18
        if trainType.find("2GPU") != -1:
            self.gpu_num = 2
        elif trainType.find("DDP") != -1:
            self.gpu_num = 4
        else:
            self.gpu_num = 1

    def set_gpu(self, tr_type):
        if tr_type.find("2GPU") != -1:
            self.gpu_num = 2
        elif tr_type.find("DDP") != -1:
            self.gpu_num = 4
        else:
            self.gpu_num = 1
    
    def _set_df(self, datafile, index_col=0, prefix="", header=0):
        data_filename = datafile

        logdir_name = data_filename.replace("_only","").replace("_prefetch1","")
        logdir_split = logdir_name.split("/")
        if logdir_split[-1].find("pcm") != -1:
            logdir_list = logdir_split[7:-1]
            # log.debug(logdir_list)
        else:
            logdir_list = logdir_split[-1].split("_")
        #['prepNloadNtrain', 'size2', 'resnet18', 'default', 'worker4', 'epoch5', 'b1024', 'cpuutil.csv']
        self.logtype = logdir_list[0]
        self.dataset = self.find_value(logdir_list, self.logtype)
        self.aug = self.find_value(logdir_list, self.dataset)
        self.model = self.find_value(logdir_list, self.aug)
        self.epoch = self.find_value(logdir_list, self.model)
        self.batchsize = self.find_value(logdir_list, self.epoch)
        self.worker = self.find_value(logdir_list, self.batchsize)
        self.thread = self.find_value(logdir_list, self.worker).replace(".csv", "")
        if not "epoch" in self.epoch:
            log.debug(f"cannot find epoch in {logdir_list}")
            return False
        self.epoch_num = int(self.epoch.replace("epoch", ""))
        if not "b" in self.batchsize:
            log.debug(f"cannot find batch in {logdir_list}")
            return False
        self.batchsize_num = int(self.batchsize.replace("b", ""))
        if not "worker" in self.worker:
            log.debug(f"cannot find worker in {logdir_list}")
            return False
        self.worker_num = int(self.worker.replace("worker", ""))
        if not "thread" in self.thread:
            log.debug(f"cannot find worker in {logdir_list}")
            return False
        self.thread_num = float(self.thread.replace("thread", ""))
        
        if trainType.find("DDPMANUAL") != -1:
            self.worker_num *= 4
        self.name = f"{prefix}{trainType}/{self.logtype}/{self.dataset}/{self.aug}/{self.model}/{self.epoch}/{self.batchsize}/{self.worker}/{self.thread}/"

        os.makedirs(savefolder+self.name, exist_ok=True)

        self.data = pd.read_csv(data_filename,
                        sep=',',
                        index_col=index_col,
                        header = header
                        )
        
        if self.data.empty:
            return False
        return True
        
    def _save_and_show(self,filename):
        # Tick setup
        plt.grid(b=True, which='major', color='#666666', linestyle=':')

        plt.savefig(savefolder+filename.replace(".csv", ""), bbox_inches='tight')
        # plt.show()

    def plot(self):
        raise NotImplementedError

    def replace_str(self, target):
        target=target.replace('\n', '')
        target=target.replace(',', '')
        return target

    def find_value(self, arr, target, jumpto=1):
        try:
            num=self.replace_str(arr[arr.index(target)+jumpto])
        except:
            raise ValueError(f'{arr}, {target}, {arr[arr.index(target)+jumpto]}')
            num='NA'
        return num

class fetchTimePlotter(basePlotter):
    def plot(self, datafile, xmin=None, xmax=None):
        if not self._set_df(datafile):
            return
        filename = f'{self.name}stall_hist_plot.svg'
        log.debug(filename)
        data = self.data[self.data["Step"] > 5]
        data = data[data["Epoch"] > 0]
        data = data[data["Epoch"] < self.epoch_num-1]
        data = data[["Training stall time (sec)"]]

        try:
            ax = data.plot.hist(bins=1000, cumulative=True, density=True)
        except:
            return
        # Figure save as

        ax.set_title(
            f"{term[self.logtype]},{term[self.aug]},{term[self.dataset]}GB {self.worker} {self.batchsize}", fontsize=self.titlesize)

        plt.axhline(y=0.9, color='r', linewidth=1.5, linestyle="--", alpha=0.75)

        ax.set_xlabel(f'{self.batchsize_num} batch fetch time (sec)')
        ax.set_ylabel('CDF')
        ax.get_legend().remove()
        ax.tick_params(direction="in")

        # ax.set_xlim(xmin=xmin, xmax=xmax)
        # Grid
        self._save_and_show(filename)

class cpuPlotter(basePlotter):
    def plot(self, datafile):
        if not self._set_df(datafile):
            return
        filename = f'{self.name}cpu_plot.svg'
        log.debug(filename)
        # try:
        data = self.data[["Socket0 Util (%)", "Socket1 Util (%)", "Total CPU Util (%)"]]
        yaxis = ["Socket0 Util (%)", "Socket1 Util (%)", "Total CPU Util (%)"]
        # except:
        #     log.debug(f"Error with data parsing.. {self.data}")
        #     return
            
        ###### Plot ######
        # -----------------
        # Line with different marker
        try:
            ax = data.plot.line(use_index=True, y=yaxis)
        except:
            return
        ax.set_title(
            f"{term[self.logtype]},{term[self.aug]},{term[self.dataset]}GB {self.worker} {self.batchsize}", fontsize=self.titlesize)

        #     ymin=0
        #     ymax=100
        #     ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('CPU Util\n(%)')

        plt.legend(fontsize=self.legendfontsize)
        plt.axhline(y=2400, color='r', linewidth=1.5, linestyle="--", alpha=0.75)
        plt.axhline(y=self.worker_num*100+100*self.gpu_num, color='r', linewidth=1.5, linestyle="-.", alpha=0.5)

        self._save_and_show(filename)

class memPlotter(basePlotter):
    def plot(self, datafile):
        if not self._set_df(datafile):
            return
        filename = f'{self.name}mem_plot.svg'
        log.debug(filename)
        data = self.data[[
        "Used memory (GB)", "Buffer/Page cache (GB)", "Swap space used (GB)"]]

        ###### Plot ######
        # -----------------
        # Line with different marker
        ax = data.plot.area()

        ax.set_title(
            f"{term[self.logtype]},{term[self.aug]},{term[self.dataset]}GB {self.worker} {self.batchsize}", fontsize=self.titlesize)

        ymin = 0
        ymax = 150
        ax.set_ylim(ymin=ymin, ymax=ymax)

        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Memory usage\n(GB)')

        plt.legend(fontsize=self.legendfontsize)
        plt.axhline(y=128, color='r', linewidth=1.5, linestyle="--", alpha=0.75)

        # Tick setup
        ax.tick_params(direction="in")

        # Grid
        plt.grid(b=True, which='major', color='#666666', linestyle=':')

        self._save_and_show(filename)

class ioPlotter(basePlotter):
    def plot(self, datafile):
        if not self._set_df(datafile):
            return
        filename = f'{self.name}io_plot.svg'
        log.debug(filename)
        self.data["MB write/s"] = self.data["KB write/s"] / 1024
        self.data["MB read/s"] = self.data["KB read/s"] / 1024
        data = self.data[["MB read/s"]]
        ###### Plot ######
        # -----------------
        # Line with different marker
        ax = data.plot.area()

        ax.set_title(
            f"{term[self.logtype]},{term[self.aug]},{term[self.dataset]}GB {self.worker} {self.batchsize}", fontsize=self.titlesize)

        ymin = 0
        ymax = 400
        ax.set_ylim(ymin=ymin, ymax=ymax)

        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Disk Usage\n(MB)')
        
        plt.legend(fontsize=self.legendfontsize)
        plt.axhline(y=390.6525, color='r', linewidth=1.5, linestyle="--", alpha=0.75)

        
        # Tick setup
        ax.tick_params(direction="in")

        self._save_and_show(filename)

class dsPlotter(basePlotter):
    def plot(self, datafile):
        self._set_df(datafile, 0)
        filename = f'{self.name}ds_plot.svg'
        log.debug(filename)
        data = self.data[["Index queues", "Worker result queue", "Data queue"]]
        yaxis = ["Index queues", "Worker result queue", "Data queue"]
        ###### Plot ######
        # -----------------
        # Line with different marker
        ax = data.plot(drawstyle="steps-post", use_index=True, y=yaxis)

        ax.set_title(
            f"{term[self.logtype]},{term[self.aug]},{term[self.dataset]}GB {self.worker} {self.batchsize}", fontsize=self.titlesize)

        ymin = -0.05

        ax.set_ylim(ymin=ymin)  # , ymax=ymax)

        ax.set_xlabel('Time')
        ax.set_ylabel('# batch data')

        plt.legend(fontsize=self.legendfontsize)

        # Tick setup
        ax.tick_params(direction="in")

        # Grid
        plt.grid(b=True, which='major', color='#666666', linestyle=':')

        self._save_and_show(filename)

class workerFetchPlotter(basePlotter):
    def plot(self, datafile):
        if not self._set_df(datafile):
            return
        filename = f'{self.name}worker_fetch_plot.svg'
        log.debug(filename)

        data = self.data
        new_data = pd.DataFrame()
        yaxis = []
        for i in range(self.gpu_num):
            try:
                gpu_data = data[data["GPU"].astype(int) == i]
            except:
                log.debug(f"Error with gpu parsing.. {data}")
                return
            
            gpu_data = gpu_data.sort_values(by=['Index number'], axis=0, ascending=True).dropna()
            # gpu_data = gpu_data.set_index('Index number')

            gpu_data=gpu_data.reset_index()
            gpu_col_name = f"GPU{i}"
            new_data[gpu_col_name] = gpu_data["Fetch time (sec)"]
            yaxis.append(gpu_col_name)
        
        ###### Plot ######
        # -----------------
        # Line with different marker
        data = new_data
        try:
            ax = data.plot.line(use_index=True, y=yaxis)
        except:
            log.debug(data)
            return

        ax.set_title(
            f"{term[self.logtype]},{term[self.aug]},{term[self.dataset]}GB {self.worker} {self.batchsize}", fontsize=self.titlesize)


        # ymin = 0
        # ymax = 1.5

        # ax.set_ylim(ymin=0)#, ymax=ymax)

        ax.set_xlabel('Index number')
        ax.set_ylabel('Fetch time (sec)')

        plt.legend(fontsize=self.legendfontsize,frameon=False)

        # Tick setup
        ax.tick_params(direction="in")

        # Grid
        plt.grid(b=True, which='major', color='#666666', linestyle=':')

        self._save_and_show(filename)


class workerFetchCdfPlotter(basePlotter):
    def plot(self, datafile):
        if not self._set_df(datafile):
            return
        filename = f'{self.name}fetch_cdf_plot.svg'
        log.debug(filename)

        data = self.data
        new_data =pd.DataFrame()
        for i in range(self.gpu_num):
            try:
                gpu_data = data[data["GPU"].astype(int) == i]
            except:
                log.debug(f"Error with gpu parsing.. {data}")
                return
            
            gpu_col_name = f"GPU{i}"
            new_data[gpu_col_name] = gpu_data["Fetch time (sec)"]
            # gpu_data = gpu_data.set_index('Index number')

        try:
            ax = new_data.plot.hist(bins=1000, cumulative=True, density=True, alpha=0.5)
        except:
            log.debug(data)
            return

        ax.set_title(
            f"{term[self.logtype]},{term[self.aug]},{term[self.dataset]}GB {self.worker} {self.batchsize}", fontsize=self.titlesize)


        # ymin = 0
        # ymax = 1.5

        # ax.set_ylim(ymin=0)#, ymax=ymax)

        ax.set_xlabel('Fetch time (sec)')
        ax.set_ylabel('CDF')

        plt.legend(fontsize=self.legendfontsize,frameon=False)

        # Tick setup
        ax.tick_params(direction="in")

        # Grid
        plt.grid(b=True, which='major', color='#666666', linestyle=':')

        ax.tick_params(direction="in")
        ax.set_xlim(xmin=0)
        self._save_and_show(filename)

   
class AugmentcdfPlotter(basePlotter):
    def plot(self, datafile, xmin=None, xmax=None):
        if not self._set_df(datafile):
            return
        filename = f'{self.name}augment_hist_plot.svg'
        log.debug(filename)

        data = self.data
        new_data =pd.DataFrame()
        for i in range(self.gpu_num):
            try:
                gpu_data = data[data["GPU"].astype(int) == i]
            except:
                log.debug(f"Error with gpu parsing.. {data}")
                return
            
            gpu_data=gpu_data.reset_index()
            gpu_col_name = f"GPU{i}"
            new_data[gpu_col_name] = gpu_data["Augmentation (sec)"]
            # gpu_data = gpu_data.set_index('Index number')

        try:
            ax = new_data.plot.hist(bins=1000, cumulative=True, density=True, alpha=0.5)
        except:
            return
        # Figure save as

        ax.set_title(
            f"{term[self.logtype]},{term[self.aug]},{term[self.dataset]}GB {self.worker} {self.batchsize}", fontsize=self.titlesize)


        ax.set_xlabel(f'Augmentation (sec)')
        ax.set_ylabel('CDF')
        plt.legend(fontsize=self.legendfontsize,frameon=False)

        ax.tick_params(direction="in")
        ax.set_xlim(xmin=0)
        if xmin is not None:
            ax.set_xlim(xmin=xmin, xmax=xmax)
        # Grid
        self._save_and_show(filename)

class delaycdfPlotter(basePlotter):
    def plot(self, datafile, xmin=None, xmax=None):
        if not self._set_df(datafile):
            return
        filename = f'{self.name}delay_cdf_plot.svg'
        log.debug(filename)
        try:
            data = self.data[self.data["Index number"] > 4]
            if self.epoch_num != 1:
                data = data[data["Epoch"] > 0]
                data = data[data["Epoch"] < self.epoch_num-1]
            log.debug(f"test with data parsing.. {data}")
            data = data[["Delayed time"]]
        except:
            log.debug(f"Error with data parsing.. {self.data}")
            return
        ###### Plot ######
        # -----------------
        # Line with different marker
        try:
            ax = data.plot.hist(bins=1000, cumulative=True, density=True)
        except:
            log.debug(f"Error with data plotting.. {self.data}")
            return
        # Figure save as

        ax.set_title(
            f"{term[self.logtype]},{term[self.aug]},{term[self.dataset]}GB {self.worker} {self.batchsize}", fontsize=self.titlesize)


        ax.set_xlabel(f'Delay time (sec)')
        ax.set_ylabel('CDF')
        ax.get_legend().remove()
        ax.tick_params(direction="in")
        ax.set_xlim(xmin=0)
        if xmin is not None:
            ax.set_xlim(xmin=xmin, xmax=xmax)
        # Grid
        self._save_and_show(filename)

class datapointPlotter(basePlotter):
    def plot(self, datafile):
        self._set_df(datafile, 1)
        filename = f'{self.name}datapoint_plot.svg'
        log.debug(filename)

        data = self.data

        yaxis = ["Get path (sec)", "Load img (sec)",
             "Decode (sec)", "Augmentation (sec)"]
        ###### Plot ######
        # -----------------
        # Line with different marker
        ax = data.plot.line(use_index=True, y=yaxis, style='x', markersize=5)

        ax.set_title(
            f"{term[self.logtype]},{term[self.aug]},{term[self.dataset]}GB {self.worker} {self.batchsize}", fontsize=self.titlesize)

        ymin = 0
        ymax = 0.20

        ax.set_ylim(ymin=ymin, ymax=ymax)

        ax.set_xlabel('Image number')
        ax.set_ylabel('Time (sec)')

        plt.legend(fontsize=self.legendfontsize,frameon=False)

        # Tick setup
        ax.tick_params(direction="in")

        self._save_and_show(filename)

class timescatterPlot(basePlotter):
    def plot(self, datafile, step_num):
        self._set_df(datafile, 0)
        filename = f'{self.name}time_scatter_plot_from_step{step_num}.svg'
        log.debug(filename)

        data = self.data

        yaxis = ["Get path (sec)", "Load img (sec)",
             "Decode (sec)", "Augmentation (sec)"]
        ###### Plot ######
        # -----------------
        # Line with different marker
        try:
            ax = data.plot(x='Step', y=[
                        'Iteration time (sec)', 'Training stall time (sec)'], style=['o', 'rx'])
        except:
            log.debug(data)
            return
        
        ax.set_title(
            f"{term[self.logtype]},{term[self.aug]},{term[self.dataset]}GB {self.worker} {self.batchsize}", fontsize=self.titlesize)

        ymin = -0.05

        ax.set_ylim(ymin=ymin)#, ymax=ymax)

        ax.set_xlabel('Steps')
        ax.set_ylabel('Time (sec)')

        plt.legend(fontsize=self.legendfontsize,frameon=False)

        # Tick setup
        ax.tick_params(direction="in")

        # Grid
        plt.grid(b=True, which='major', color='#666666', linestyle=':')

        self._save_and_show(filename)
        
class iterPlotter(basePlotter):
    def plot(self, datafile):
        if not self._set_df(datafile):
            return
        filename = f'{self.name}iter_plot.svg'
        log.debug(filename)
        try:
            data = self.data[self.data["Step"] > 4]
            data = data[data["Epoch"] > 0]
            data = data[data["Epoch"] < self.epoch_num-1]
            data = data[["Iteration time (sec)", "Training stall time (sec)"]]
        except:
            log.debug(f"Error with data parsing.. {self.data}")
            return
        
        try:
            data["Time"] = data["Time"].dt.total_seconds()
        except:
            log.debug(f"There is no time in {data}")
            return
        data["Time"] = data["Time"] - data["Time"].iloc[0]
        yaxis = ["Iteration time (sec)", "Training stall time (sec)"]

        ###### Plot ######
        # -----------------
        # Line with different marker
        try:
            ax = data.plot.line(use_index=True, y=yaxis, linewidth = 0.0 ,style=['.', 'x'])
        except:
            log.debug(f"Error with data plotting {data}")
            return
        ax.set_title(
            f"{term[self.logtype]},{term[self.aug]},{term[self.dataset]}GB {self.worker} {self.batchsize}", fontsize=self.titlesize)

        # ymin=0
        # #     ymax=100
        # ax.set_ylim(ymin=ymin)#, ymax=ymax)
        ax.set_xlabel('Elapsed time (sec)')
        ax.set_ylabel('Time (sec)')

        plt.legend(fontsize=self.legendfontsize)
        # plt.axhline(y=2400, color='r', linewidth=1.5, linestyle="--", alpha=0.75)
        # plt.axhline(y=self.worker_num*100+100*self.gpu_num, color='r', linewidth=1.5, linestyle="-.", alpha=0.5)

        self._save_and_show(filename)

class gpuiterPlotter(basePlotter):
    def plot(self, datafile):
        if not self._set_df(datafile):
            return
        filename = f'{self.name}iter_gpu_plot.svg'
        
        try:
            data = self.data[self.data["Step"] > 4].dropna()
            data = data[data["Epoch"] > 0]
            data = data[data["Epoch"] < self.epoch_num-1]
            yaxis = []
        except:
            log.debug(f"Error with Parsing {self.data}")
            return
        # only GPU iteration (sec)
        new_data = pd.DataFrame()
        for i in range(self.gpu_num):
            try:
                gpu_data = data[data["GPU"].astype(int) == i]
            except:
                return
            
            gpu_col_name = f"GPU{i}"
            new_data[gpu_col_name] = gpu_data["Iteration time (sec)"]
            yaxis.append(gpu_col_name)
        try:
            data["Time"] = data["Time"].dt.total_seconds()
        except:
            log.debug(f"There is no time in {data}")
            return
        new_data["Time"] = data["Time"] - data["Time"].iloc[0]
        new_data = new_data.set_index('Time')
        # log.debug(new_data)

        data = new_data
        # except:
        #     log.debug(f"Error with data parsing.. {self.data}")
        #     return
        

        ###### Plot ######
        # -----------------
        # Line with different marker
        try:
            ax = data.plot.line(use_index=True, y=yaxis, linewidth = 0.0 ,style=['.', 'x', 's', '^'])
        except:
            log.debug(f"Error with data plotting {data}")
            return
        ax.set_title(
            f"{term[self.logtype]},{term[self.aug]},{term[self.dataset]}GB {self.worker} {self.batchsize}", fontsize=self.titlesize)

        # ymin=0
        # #     ymax=100
        # ax.set_ylim(ymin=ymin)#, ymax=ymax)
        ax.set_xlabel('#th iteration')
        ax.set_ylabel('Elapsed time (sec)')

        plt.legend(fontsize=self.legendfontsize)
        # plt.axhline(y=2400, color='r', linewidth=1.5, linestyle="--", alpha=0.75)
        # plt.axhline(y=self.worker_num*100+100*self.gpu_num, color='r', linewidth=1.5, linestyle="-.", alpha=0.5)
        log.debug(filename)
        self._save_and_show(filename)
        
class itercdfPlotter(basePlotter):
    def plot(self, datafile, xmin=None, xmax=None):
        if not self._set_df(datafile):
            log.debug(f"{datafile} cannot load")
            return
        filename = f'{self.name}iter_hist_plot.svg'
        log.debug(filename)
        try:
            data = self.data[self.data["Step"] > 4]
            data = data[data["Epoch"] > 0]
            data = data[data["Epoch"] < self.epoch_num-1]
            data = data[["Iteration time (sec)"]]
        except:
            log.debug(f"Error with data parsing.. {self.data}")
            return

        ###### Plot ######
        # -----------------
        # Line with different marker
        try:
            ax = data.plot.hist(bins=1000, cumulative=True, density=True)
        except:
            return
        # Figure save as

        ax.set_title(
            f"{term[self.logtype]},{term[self.aug]},{term[self.dataset]}GB {self.worker} {self.batchsize}", fontsize=self.titlesize)


        ax.set_xlabel(f'Iteration time (sec)')
        ax.set_ylabel('CDF')
        ax.get_legend().remove()
        ax.tick_params(direction="in")
        ax.set_xlim(xmin=0)
        if xmin is not None:
            ax.set_xlim(xmin=xmin, xmax=xmax)
        # Grid
        self._save_and_show(filename)

class delaycdfPlotter(basePlotter):
    def plot(self, datafile, xmin=None, xmax=None):
        if not self._set_df(datafile):
            return
        filename = f'{self.name}delay_cdf_plot.svg'
        log.debug(filename)
        if self.data.empty:
            return
        try:
            data = self.data[self.data["Index number"] > 4]
            if self.epoch_num != 1:
                data = data[data["Epoch"] > 0]
                data = data[data["Epoch"] < self.epoch_num-1]
            log.debug(f"test with data parsing.. {data}")
            data = data[["Delayed time"]]
        except:
            log.debug(f"Error with data parsing.. {self.data}")
            return
        ###### Plot ######
        # -----------------
        # Line with different marker
        try:
            ax = data.plot.hist(bins=1000, cumulative=True, density=True)
        except:
            log.debug(f"Error with data plotting.. {self.data}")
            return
        # Figure save as

        ax.set_title(
            f"{term[self.logtype]},{term[self.aug]},{term[self.dataset]}GB {self.worker} {self.batchsize}", fontsize=self.titlesize)


        ax.set_xlabel(f'Delay time (sec)')
        ax.set_ylabel('CDF')
        ax.get_legend().remove()
        ax.tick_params(direction="in")
        ax.set_xlim(xmin=0)
        if xmin is not None:
            ax.set_xlim(xmin=xmin, xmax=xmax)
        # Grid
        self._save_and_show(filename)

class batchLifeTimePlotter(basePlotter):
    def plot(self, datafile):
        if not self._set_df(datafile):
            return
        filename = f'{self.name}lifetime.svg'
        if self.data.empty:
            # log.debug(f"{filename} is empty")
            return
        # try:
        # data = self.data[self.data["Index number"] > 4].dropna()
        data = self.data 
        yaxis = []
        
        # only GPU iteration (sec)
        new_data = pd.DataFrame()
        for i in range(self.gpu_num):
            try:
                gpu_data = data[data["GPU"].astype(int) == i]
            except:
                log.debug(f"Error with gpu parsing.. {data}")
                return
            
            gpu_data = gpu_data.sort_values(by=['Index number'], axis=0, ascending=True).dropna()
            gpu_data = gpu_data[gpu_data["Index number"] > 10]
            gpu_data = gpu_data.set_index('Index number')

            gpu_col_name = f"GPU{i}"
            new_data[gpu_col_name] = gpu_data["Batch life time (sec)"]
            yaxis.append(gpu_col_name)
        
        # log.debug(new_data)
        # except:
        #     log.debug(f"Error with data parsing.. {self.data}")
        #     return

        ###### Plot ######
        # -----------------
        # Line with different marker

        try:
            ax = new_data.plot.line(use_index=True, y=yaxis, linewidth=1.5, alpha=0.8)
        #     ax = data.plot.hist(bins=1000, cumulative=True, density=True, alpha=0.5)
        # ax = data.plot.hist(bins=1000, cumulative=True, density=True, alpha=0.5)
        except:
            log.debug(f"Error with data plotting {data}")
            return
        
        ax.set_title(
            f"{term[self.logtype]},{term[self.aug]},{term[self.dataset]}GB {self.worker} {self.batchsize}", fontsize=self.titlesize)

        # ymin=0
        # # ymax=0.6
        # ax.set_ylim(ymin=ymin)#,ymax=ymax)
        # xmin = 0
        
        ax.set_xlabel(f'Index number')
        ax.set_ylabel('Batch life time (sec)')

        plt.legend(fontsize=self.legendfontsize)

        ax.tick_params(direction="in")
        ax.set_xlim(xmin=0)
        
        # Grid
        self._save_and_show(filename)


class gpuitercdfPlotter(basePlotter):
    def plot(self, datafile):
        if not self._set_df(datafile):
            return
        filename = f'{self.name}iter_gpucdf_plot.svg'

        if self.data.empty:
            # log.debug(f"{filename} is empty")
            return

        try:
            data = self.data[self.data["Step"] > 4].dropna()
            data = data[data["Epoch"] > 0]
            data = data[data["Epoch"] < self.epoch_num-1]
        except:
            log.debug(f"Error in {self.data}")
            return
        yaxis = []
        
        # only GPU iteration (sec)
        new_data = pd.DataFrame()
        for i in range(self.gpu_num):
            try:
                gpu_data = data[data["GPU"].astype(int) == i]
            except:
                return
            gpu_data = gpu_data.reset_index()
            gpu_col_name = f"GPU{i}"
            new_data[gpu_col_name] = gpu_data["Iteration time (sec)"]
            yaxis.append(gpu_col_name)
        # log.debug(new_data)
        # except:
        #     log.debug(f"Error with data parsing.. {self.data}")
        #     return
        

        ###### Plot ######
        # -----------------
        # Line with different marker
        # try:
        try:
            ax = new_data[yaxis].hist(bins=1000, cumulative=True, density=True, alpha=0.5)
        except:
            log.debug(f"Error to plot with {new_data}")
            return
        # ax = data.plot.hist(bins=1000, cumulative=True, density=True, alpha=0.5)
        # except:
        #     log.debug(f"Error with data plotting {data}")
        #     return
        
        ax.set_title(
            f"{term[self.logtype]},{term[self.aug]},{term[self.dataset]}GB {self.worker} {self.batchsize}", fontsize=self.titlesize)

        # ymin=0
        # #     ymax=100
        # ax.set_ylim(ymin=ymin)#, ymax=ymax)
        xmin = 0
        
        ax.set_xlabel(f'Iteration time (sec)')
        ax.set_ylabel('CDF')
        # ax.get_legend().remove()
        ax.tick_params(direction="in")
        ax.set_xlim(xmin=0)
        
        # Grid
        self._save_and_show(filename)

class ipcPlotter(basePlotter):
    def plot(self, datafile):
        self._set_df(datafile, prefix = "pcm/", index_col=None, header = [0,1])
        filename = f'{self.name}pcm_IPC.svg'
        log.debug(filename)
        try:
            data = self.data
            data.columns = data.columns.map('_'.join)
            data = data[["System_IPC", "Socket 0_IPC", "Socket 1_IPC"]]
        except:
            log.debug(f"Error with data parsing.. {self.data}")
            return
        
        yaxis = ["System_IPC", "Socket 0_IPC", "Socket 1_IPC"]

        ###### Plot ######
        # -----------------
        # Line with different marker
        try:
            ax = data.plot.line(use_index=True, y=yaxis)
        except:
            log.debug(f"Error with data plotting {data}")
            return
        ax.set_title(
            f"{term[self.logtype]},{term[self.aug]},{term[self.dataset]}GB {self.worker} {self.batchsize}", fontsize=self.titlesize)

        ymin=0
        ymax=2.5
        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('IPC')

        plt.legend(fontsize=self.legendfontsize)
        # plt.axhline(y=2400, color='r', linewidth=1.5, linestyle="--", alpha=0.75)
        # plt.axhline(y=self.worker_num*100+100*self.gpu_num, color='r', linewidth=1.5, linestyle="-.", alpha=0.5)

        self._save_and_show(filename)
        
class l3missPlotter(basePlotter):
    def plot(self, datafile):
        self._set_df(datafile, prefix = "pcm/", index_col=None, header = [0,1])
        filename = f'{self.name}pcm_L3miss.svg'
        log.debug(filename)
        try:
            data = self.data
            data.columns = data.columns.map('_'.join)
            data = data[["System_L3MISS", "Socket 0_L3MISS", "Socket 1_L3MISS"]]
        except:
            log.debug(f"Error with data parsing.. {self.data}")
            return
        
        yaxis = ["System_L3MISS", "Socket 0_L3MISS", "Socket 1_L3MISS"]

        ###### Plot ######
        # -----------------
        # Line with different marker
        try:
            ax = data.plot.line(use_index=True, y=yaxis)
        except:
            log.debug(f"Error with data plotting {data}")
            return
        ax.set_title(
            f"{term[self.logtype]},{term[self.aug]},{term[self.dataset]}GB {self.worker} {self.batchsize}", fontsize=self.titlesize)

        ymin=0
        ymax=60
        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('L3 misses\n(Millions)')

        plt.legend(fontsize=self.legendfontsize)
        # plt.axhline(y=2400, color='r', linewidth=1.5, linestyle="--", alpha=0.75)
        # plt.axhline(y=self.worker_num*100+100*self.gpu_num, color='r', linewidth=1.5, linestyle="-.", alpha=0.5)

        self._save_and_show(filename)
        
class l3hitPlotter(basePlotter):
    def plot(self, datafile):
        self._set_df(datafile, prefix = "pcm/", index_col=None, header = [0,1])
        filename = f'{self.name}pcm_L3hit.svg'
        log.debug(filename)
        try:
            data = self.data
            data.columns = data.columns.map('_'.join)
            data = data[["System_L3HIT", "Socket 0_L3HIT", "Socket 1_L3HIT"]]
        except:
            log.debug(f"Error with data parsing.. {self.data}")
            return
        
        yaxis = ["System_L3HIT", "Socket 0_L3HIT", "Socket 1_L3HIT"]
        
        ###### Plot ######
        # -----------------
        # Line with different marker
        try:
            ax = data.plot.line(use_index=True, y=yaxis)
        except:
            log.debug(f"Error with data plotting {data}")
            return
        ax.set_title(
            f"{term[self.logtype]},{term[self.aug]},{term[self.dataset]}GB {self.worker} {self.batchsize}", fontsize=self.titlesize)

        ymin=0
        ymax=1.0
        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('L3 hit ratio\n(hits/reference)')

        plt.legend(fontsize=self.legendfontsize)
        # plt.axhline(y=2400, color='r', linewidth=1.5, linestyle="--", alpha=0.75)
        # plt.axhline(y=self.worker_num*100+100*self.gpu_num, color='r', linewidth=1.5, linestyle="-.", alpha=0.5)

        self._save_and_show(filename)

perf_dict = {
    "0":"l1d.replacement",
    "1":"l1d_pend_miss.fb_full",
    "2":"l2_rqsts.miss",
    "3":"l2_rqsts.references",
    "4":"longest_lat_cache.miss",
    "5":"longest_lat_cache.reference",
    "6":"offcore_requests.l3_miss_demand_data_rd",
    "7":"offcore_requests_outstanding.l3_miss_demand_data_rd",
    "8":"offcore_requests_outstanding.all_data_rd",
    "9":"offcore_requests_buffer.sq_full",
    "10":"dtlb_load_misses.miss_causes_a_walk",
    "11":"dtlb_store_misses.miss_causes_a_walk",
}

class perfhistPlotter(basePlotter):
    def plot(self, datafile):
        self.data = pd.read_csv(datafile,
                        sep=',',
                        index_col=None,
                        header = None
                        )
        df_name = ["None","Percent","Program","Library","User/Kernel space","Symbol"]
        self.data.columns = df_name    
        # log.debug(datafile.split("."))
        perf_num = datafile.split(".")[-1]
        # log.debug(f"perf_num: {perf_num}")
        perf_type = perf_dict[perf_num]
        # log.debug(f"perf_type: {perf_type}")
        # return 
        
        filename = f'/perf/{trainType}/perf_{perf_type}.svg'
        log.debug(filename)
        
        yaxis = ["Library", "Symbol", "Percent"]
        
        try:
            data = self.data[yaxis]
        except:
            log.debug(f"Error with data parsing.. {self.data}")
            return
        
        
        ###### Plot ######
        # -----------------
        # Line with different marker
        try:
            fig, ax = plt.subplots()
            ax.pie(data.groupby('Library', sort=False)['Percent'].sum(), 
                   radius=1, labels=data['Library'].drop_duplicates(), autopct='%1.2f%%',
                    textprops={'fontsize': 18}, wedgeprops=dict(width=0.3, edgecolor='w'))

        except:
            log.debug(f"Error with data plotting {data}")
            return
        ax.set_title(
            f"{trainType}, {perf_type}", fontsize=self.titlesize)

        # ymin=0
        # ymax=1.0
        # ax.set_ylim(ymin=ymin, ymax=ymax)
        # ax.set_xlabel('Time (sec)')
        # ax.set_ylabel('L3 hit ratio\n(hits/reference)')
        ax.legend().set_visible(False)
        # plt.legend(fontsize=self.legendfontsize, bbox_to_anchor=(1.2, 1.2))
        # plt.axhline(y=2400, color='r', linewidth=1.5, linestyle="--", alpha=0.75)
        # plt.axhline(y=self.worker_num*100+100*self.gpu_num, color='r', linewidth=1.5, linestyle="-.", alpha=0.5)

        self._save_and_show(filename)

def plotWrapper(typedir='', file_suffix='', parser=None, plotter=None, extension_suffix=".csv"):
    for tr_type in parsePackage.trainTypeSet:
        parsePackage.redefine(tr_type)
        set_dir(tr_type)
        
        if parser is not None:
            parser.parser()
        
        data_filedir = parsed_dir + f'/{typedir}/'
        datafiles = glob.glob(f"{data_filedir}*{file_suffix}{extension_suffix}")

        log.debug(tr_type)
        log.debug(datafiles)
        
        if plotter is None:
            return
        
        log.debug("Plotting...")
        plotter.set_gpu(tr_type)
        
        with Pool() as p:
            p.map(plotter.plot, datafiles)
