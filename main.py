import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import pdb
import sys
import cv2
import yaml
import torch
import random
import importlib
import faulthandler
import numpy as np
import torch.nn as nn
import shutil
import inspect
import time
from collections import OrderedDict
import wandb
import torch.optim as optim
import gc
from CorrNet.CorrNet.CPC import CPC, ResNet_LSTM

faulthandler.enable()
import utils
from modules.sync_batchnorm import convert_model
from seq_scripts import seq_train, seq_eval, seq_feature_generation
from torch.cuda.amp import autocast as autocast
from cpc_train import cpc_pretrain, train_with_cpc



class Processor():
    def __init__(self, arg):


        self.arg = arg
        if os.path.exists(self.arg.work_dir):
            answer = input('Current dir exists, do you want to remove and refresh it?\n')
            if answer in ['yes','y','ok','1']:
                print('Dir removed !')
                shutil.rmtree(self.arg.work_dir)
                os.makedirs(self.arg.work_dir)
            else:
                print('Dir Not removed !')
        else:
            os.makedirs(self.arg.work_dir)
        shutil.copy2(__file__, self.arg.work_dir)
        shutil.copy2('./configs/baseline.yaml', self.arg.work_dir)
        shutil.copy2('./modules/tconv.py', self.arg.work_dir)
        shutil.copy2('./modules/resnet.py', self.arg.work_dir)
        self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval)
        self.save_arg()
        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)
        self.device = utils.GpuDataParallel()
        self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval)
        self.dataset = {}
        self.data_loader = {}
        self.gloss_dict = np.load(self.arg.dataset_info['dict_path'], allow_pickle=True).item()
        self.arg.model_args['num_classes'] = len(self.gloss_dict) + 1
        self.model, self.optimizer = self.loading()


    def start(self):
        print(f"PHASE IS {self.arg.phase}")
        if self.arg.phase == 'train':
            wandb.init(
            # set the wandb project where this run will be logged
            project="COMP9491",
            name = "Resnet corr + Lin+BiLSTMs",
            # track hyperparameters and run metadata
            config={
                "architecture": "pretrain ",
                "dataset": "PhonenixT",
                "epochs": self.arg.num_epoch,
                }
            )
            
            
            best_dev = 100.0
            best_epoch = 0
            total_time = 0
            epoch_time = 0
            self.recoder.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            seq_model_list = []
            for epoch in range(self.arg.optimizer_args['start_epoch'], self.arg.num_epoch):
                save_model = epoch % self.arg.save_interval == 0
                eval_model = epoch % self.arg.eval_interval == 0
                epoch_time = time.time()
                # train end2end model
                loss_mean=seq_train(self.data_loader['train'], self.model, self.optimizer,
                          self.device, epoch, self.recoder, compute_cpc = False)
                if eval_model:
                    dev_wer = seq_eval(self.arg, self.data_loader['dev'], self.model, self.device,
                                       'dev', epoch, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
                    self.recoder.print_log("Dev WER: {:05.2f}%".format(dev_wer))
                # if dev_wer < best_dev:
                #     best_dev = dev_wer
                #     best_epoch = epoch
                #     model_path = "{}_best_model.pt".format(self.arg.work_dir)
                #     self.save_model(epoch, model_path)
                #     self.recoder.print_log('Save best model')
                self.recoder.print_log('Best_dev: {:05.2f}, Epoch : {}'.format(best_dev, best_epoch))
                if save_model:
                    model_path = "{}dev_{:05.2f}_epoch{}_model.pt".format(self.arg.work_dir, dev_wer, epoch)
                    seq_model_list.append(model_path)
                    print("seq_model_list", seq_model_list)
                    self.save_model(epoch, model_path)

                epoch_time = time.time() - epoch_time
                total_time += epoch_time
                torch.cuda.empty_cache()
                
                wandb.log({"dev_wer": dev_wer, "loss": loss_mean})
                
                self.recoder.print_log('Epoch {} costs {} mins {} seconds'.format(epoch, int(epoch_time)//60, int(epoch_time)%60))
            self.recoder.print_log('Training costs {} hours {} mins {} seconds'.format(int(total_time)//60//60, int(total_time)//60%60, int(total_time)%60))
        elif self.arg.phase == 'test':
            if self.arg.load_weights is None and self.arg.load_checkpoints is None:
                print('Please appoint --weights.')
            self.recoder.print_log('Model:   {}.'.format(self.arg.model))
            self.recoder.print_log('Weights: {}.'.format(self.arg.load_weights))
            # train_wer = seq_eval(self.arg, self.data_loader["train_eval"], self.model, self.device,
            #                      "train", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
            
            
            # dev_wer = seq_eval(self.arg, self.data_loader["dev"], self.model, self.device,
            #                    "dev", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
            test_wer = seq_eval(self.arg, self.data_loader["test"], self.model, self.device,
                                "test", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
            self.recoder.print_log('Evaluation Done.\n')
        elif self.arg.phase == "features":
            for mode in ["train", "dev", "test"]:
                seq_feature_generation(
                    self.data_loader[mode + "_eval" if mode == "train" else mode],
                    self.model, self.device, mode, self.arg.work_dir, self.recoder
                )
                
        elif self.arg.phase == 'CPC_pretrain':
            wandb.init(
            # set the wandb project where this run will be logged
            project="COMP9491-CPC",

                # track hyperparameters and run metadata
                config={
                    "architecture": "pretrain ",
                    "dataset": "PhonenixT",
                    "epochs": self.arg.num_epoch,
                    }
            )
            # cpc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # cpc_device = self.device.get_device()
           # print(f"IIN CPC TRIANING {self.device}")
           # print(f"OUTPUT DEVICE IS {self.device.output_device} and num gpu is {len(self.device.gpu_list)}")
            self.model = ResNet_LSTM(n_pred_step = 4,
               context_dim = 512, 
               embedding_dim = 512,
               hidden_dim = 512,
               num_layers = 3, 
               device = self.device)
            self.model = self.model_to_device(self.model)
            self.cpc_op = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
            best_loss = 100
            best_epoch = 0
            total_epoch = 30
            for epoch in range(total_epoch):
                # print("Running")
                loss,acc= cpc_pretrain(self.data_loader['train'],self.model,self.cpc_op, self.device, epoch)
                
                wandb.log({'loss':loss,'accuracy':acc})
                # if loss < best_loss:
                #     best_loss = loss
                #     best_epoch = epoch
                # torch.cuda.empty_cache()
                if (epoch+1) % 2 == 0:
                    # pretrain model save
                    file_name = "epoch{}_bs{}_model.pt".format(str(epoch+1),str(self.arg.batch_size))
                    path = os.path.join(self.arg.pretrain_CPC_path,file_name)
                    torch.save(self.model, path)
                    print(f"Model saved at epoch = {epoch+1}")
                del loss
                del acc

                # gc.collect()
                torch.cuda.empty_cache()
        elif self.arg.phase == 'CPC_train':
            load_model = True
            
            wandb.init(
            # set the wandb project where this run will be logged
                project="COMP9491",
                name = "Use context No CPC pretrain,Origiinal loss ",
                # track hyperparameters and run metadata
                config={
                    "architecture": "1D pretrain COC (ResNet Freeze) ",
                    "dataset": "PhonenixT",
                    "epochs": self.arg.num_epoch,
                    }
            )
            
            # path = os.path.join(self.arg.pretrain_CPC_path,"epoch1_model.pt")
            if load_model is True:
                if os.path.exists(self.arg.train_CPC_load_path):

                    print(f"path to load is {self.arg.train_CPC_load_path}")
                    loaded_cpc = torch.load(self.arg.train_CPC_load_path)
                    loaded_cpc.initial_extractor.requires_grad_(True)
                    loaded_cpc.train()
                    loaded_cpc.requires_grad_(True) 

                else:
                    raise ValueError("Path to model not exists")
                #    Loaded CPC parameters
                # for param in loaded_cpc.parameters():
                #     param.requires_grad = False
            else:
                loaded_cpc = CPC(n_pred_step = 2,
               context_dim = 512, 
               embedding_dim = 512,
               hidden_dim = 512,
               num_layers = 3, 
               device = self.device) # n_pred_step set to 2 or 4
                print(f"Sending the loaded cpc to device!!!!!!!!!!!!!!!!!!")
            loaded_cpc = self.model_to_device(loaded_cpc)
            self.model.set_cpc(loaded_cpc,"embedding") # embedding, context

            for epoch in range(0, self.arg.num_epoch):
                # train end2end model
                # loss_mean = train_with_cpc(self.data_loader['train'], self.model, self.optimizer,
                #          self.device, epoch, self.recoder)
                print(f"At epoch {epoch}")
                loss_mean=seq_train(self.data_loader['train'], self.model, self.optimizer,
                          self.device, epoch, self.recoder, compute_cpc = False)
                dev_wer = seq_eval(self.arg, self.data_loader['dev'], self.model, self.device,
                                       'dev', epoch, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
                self.recoder.print_log("Dev WER: {:05.2f}%".format(dev_wer))
                # wandb.init("
                wandb.log({"dev_wer": dev_wer, "loss": loss_mean})
                torch.cuda.empty_cache()
            print(f"Done")
        elif self.arg.phase == 'ResNet_LSTM':
            wandb.init(
            # set the wandb project where this run will be logged
                project="COMP9491",
                name = "only ResNet with Corr to feature extract ",
                # track hyperparameters and run metadata
                config={
                    "architecture": "Use ResNet-LSTM followed by the 1D COnv ",
                    "dataset": "PhonenixT",
                    "epochs": self.arg.num_epoch,
                    }
            )
            res_lstm = ResNet_LSTM(n_pred_step = 4,
               context_dim = 512, 
               embedding_dim = 512,
               hidden_dim = 512,
               num_layers = 3, 
               device = self.device)
            res_lstm = self.model_to_device(res_lstm)
            self.model.set_res_lstm(res_lstm)
            for epoch in range(0, self.arg.num_epoch):
                loss_mean = train_with_cpc(self.data_loader['train'], self.model, self.optimizer,
                          self.device, epoch, self.recoder)
            # train end2end model
                dev_wer = seq_eval(self.arg, self.data_loader['dev'], self.model, self.device,
                                       'dev', epoch, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
                self.recoder.print_log("Dev WER: {:05.2f}%".format(dev_wer))
                # wandb.init("
                wandb.log({"dev_wer": dev_wer, "loss": loss_mean})
                torch.cuda.empty_cache()
                
                

            
        

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def save_model(self, epoch, save_path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
            'rng_state': self.rng.save_rng_state(),
        }, save_path)

    def loading(self):
        print(f"Arg device is {self.arg.device}")
        self.device.set_device(self.arg.device)
        print("Loading model")
        model_class = import_class(self.arg.model)
        model = model_class(
            **self.arg.model_args,
            gloss_dict=self.gloss_dict,
            loss_weights=self.arg.loss_weights,
        )
        shutil.copy2(inspect.getfile(model_class), self.arg.work_dir)
        optimizer = utils.Optimizer(model, self.arg.optimizer_args)

        if self.arg.load_weights:
            self.load_model_weights(model, self.arg.load_weights)
        elif self.arg.load_checkpoints:
            self.load_checkpoint_weights(model, optimizer)
        
        #self.kernel_sizes = model.module.conv1d.kernel_size
        self.kernel_sizes = model.conv1d.kernel_size
        model = self.model_to_device(model)
        print("Loading model finished.")
        self.load_data()
        return model, optimizer

    def model_to_device(self, model):
        model = model.to(self.device.output_device)
        print(len(self.device.gpu_list))
        # if len(self.device.gpu_list) > 1:
        #     raise ValueError("AMP equipped with DataParallel has to manually write autocast() for each forward function, you can choose to do this by yourself")
            #model.conv2d = nn.DataParallel(model.conv2d, device_ids=self.device.gpu_list, output_device=self.device.output_device)
        
        model = convert_model(model)
        # if len(self.device.gpu_list) > 1:
        #     model = nn.parallel.DataParallel(
        #         model,device_ids=self.device.gpu_list)
        model.cuda()

        return model

        
#     def model_to_device(model,device):
#         print(f"MMODEL IS TO {device.output_device} GPU LIst {device.gpu_list}")
#         #model = model.cuda(self.device.output_device)
#         model = model.to(device.output_device)

#         if len(self.device.gpu_list) > 1:
#             model = nn.DataParallel(
#                 model,
#                 device_ids=self.device.gpu_list)
#         return model

    def load_model_weights(self, model, weight_path):
        state_dict = torch.load(weight_path)
        if len(self.arg.ignore_weights):
            for w in self.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print('Successfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
        weights = self.modified_weights(state_dict['model_state_dict'], False)
        # weights = self.modified_weights(state_dict['model_state_dict'])
        model.load_state_dict(weights, strict=True)

    @staticmethod
    def modified_weights(state_dict, modified=False):
        state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
        if not modified:
            return state_dict
        modified_dict = dict()
        return modified_dict

    def load_checkpoint_weights(self, model, optimizer):
        self.load_model_weights(model, self.arg.load_checkpoints)
        state_dict = torch.load(self.arg.load_checkpoints)

        if len(torch.cuda.get_rng_state_all()) == len(state_dict['rng_state']['cuda']):
            print("Loading random seeds...")
            self.rng.set_rng_state(state_dict['rng_state'])
        if "optimizer_state_dict" in state_dict.keys():
            print("Loading optimizer parameters...")
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            optimizer.to(self.device.output_device)
        if "scheduler_state_dict" in state_dict.keys():
            print("Loading scheduler parameters...")
            optimizer.scheduler.load_state_dict(state_dict["scheduler_state_dict"])

        self.arg.optimizer_args['start_epoch'] = state_dict["epoch"] + 1
        self.recoder.print_log("Resuming from checkpoint: epoch {self.arg.optimizer_args['start_epoch']}")

    def load_data(self):
        print("Loading data")
        self.feeder = import_class(self.arg.feeder)
        shutil.copy2(inspect.getfile(self.feeder), self.arg.work_dir)
        if self.arg.dataset == 'CSL':
            dataset_list = zip(["train", "dev"], [True, False])
        elif 'phoenix' in self.arg.dataset:
            dataset_list = zip(["train", "train_eval", "dev", "test"], [True, False, False, False]) 
        elif self.arg.dataset == 'CSL-Daily':
            dataset_list = zip(["train", "train_eval", "dev", "test"], [True, False, False, False])
        for idx, (mode, train_flag) in enumerate(dataset_list):
            arg = self.arg.feeder_args
            arg["prefix"] = self.arg.dataset_info['dataset_root']
            arg["mode"] = mode.split("_")[0]
            arg["transform_mode"] = train_flag
            self.dataset[mode] = self.feeder(gloss_dict=self.gloss_dict, kernel_size= self.kernel_sizes, dataset=self.arg.dataset, **arg)
            self.data_loader[mode] = self.build_dataloader(self.dataset[mode], mode, train_flag)
        print("Loading data finished.")
    def init_fn(self, worker_id):
        np.random.seed(int(self.arg.random_seed)+worker_id)
    def build_dataloader(self, dataset, mode, train_flag):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.arg.batch_size if mode == "train" else self.arg.test_batch_size,
            shuffle=train_flag,
            drop_last=train_flag,
            num_workers=self.arg.num_worker,  # if train_flag else 0
            collate_fn=self.feeder.collate_fn,
            pin_memory=True,
            worker_init_fn=self.init_fn,
        )


def import_class(name):
    components = name.rsplit('.', 1)
    
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod


if __name__ == '__main__':
    
    count = torch.cuda.device_count()
    print(count)
    for i in range(count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    sparser = utils.get_parser()
    p = sparser.parse_args()
    # p.config = "baseline_iter.yaml"
    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
        
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                print(key)
                assert (k in key)
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()
    with open(f"./configs/{args.dataset}.yaml", 'r') as f:
        
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)
        print(args)
    processor = Processor(args)
    utils.pack_code("./", args.work_dir)
    processor.start()