import torch
from torch._C import ThroughputBenchmark
from torch.autograd.grad_mode import no_grad
from torchvision import models
import sys, os
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models.resnet import resnet18
import dataset
import argparse
from operator import itemgetter
import time
import tensorly as tl
import tensorly
from itertools import chain
from decompositions import cp_decomposition_conv_layer, tensor_train_decomposition_conv_layer, tucker_decomposition_conv_layer
from pytorch_utils.models import initialize_model
from pthflops import count_ops



class Trainer:
    def __init__(self, train_path, test_path, model, optimizer, batch_size=64):
        self.train_data_loader = dataset.loader(train_path, batch_size=batch_size)
        self.test_data_loader = dataset.test_loader(test_path, batch_size=batch_size)

        self.optimizer = optimizer

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model.train()

    def test(self):
        self.model.cuda()
        self.model.eval()
        correct = 0
        total = 0
        total_time = 0
        for i, (batch, label) in enumerate(self.test_data_loader):
            batch = batch.cuda()
            t0 = time.time()
            output = model(batch).cpu()
            t1 = time.time()
            total_time = total_time + (t1 - t0)
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)
        
        print("Accuracy :", float(correct) / total)
        print("Average prediction time", float(total_time) / (i + 1), i + 1)

        self.model.train()

    def train(self, epoches=10):
        for i in range(epoches):
            print("Epoch: ", i)
            self.train_epoch()
            self.test()
        print("Finished fine tuning.")
        

    def train_batch(self, batch, label):
        # print(batch.shape)
        self.model.zero_grad()
        input = batch
        self.criterion(self.model(input), label).backward()
        self.optimizer.step()

    def train_epoch(self):
        for i, (batch, label) in enumerate(self.train_data_loader):
            self.train_batch(batch.cuda(), label.cuda())

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--decompose", dest="decompose", action="store_true")
    parser.add_argument("--fine_tune", dest="fine_tune", action="store_true")
    parser.add_argument("--profile", dest="profile", action="store_true")
    parser.add_argument("--flops", dest="flops", action="store_true")
    parser.add_argument("--fw", dest="fw", action="store_true")
    parser.add_argument("--snap",type=str,default="")
    parser.add_argument("--all_layers", dest="all_layers", action="store_true")
    parser.add_argument("--train_path", type = str, default = "train")
    parser.add_argument("--test_path", type = str, default = "test")
    parser.add_argument("--cp", dest="cp", action="store_true", \
        help="Use cp decomposition. uses tucker by default")
    parser.add_argument("--tt", dest="tt", action="store_true", \
        help="Use tensor train decomposition. uses tucker by default")
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(train=False)
    parser.set_defaults(decompose=False)
    parser.set_defaults(fine_tune=False)
    parser.set_defaults(cp=False)
    parser.add_argument("--batch_size", type = int, default = 64)    
    args = parser.parse_args()
    return args


def decompose(parent:torch.nn.Module,parent_name:str, ancestors:str=""):
    # leaf node was reached, end recursion
    if parent is None:
        return
    
    # track ancestors
    if ancestors == "":
        ancestors = parent_name
    else:
        ancestors = ancestors + "." + parent_name


    children = parent.named_children()
    #iterate over layers
    for name, child in children:
        # found conv layer
        if isinstance(child, torch.nn.modules.conv.Conv2d):
            # dont decompose early layers OR downsample
            if (not args.all_layers and child.in_channels<65) or child.kernel_size==(1,1):
                continue
            print("decomposing %s.%s"%(ancestors, name))
            
            if args.cp:
                rank = max(child.weight.data.numpy().shape)//2
                setattr(parent, name, cp_decomposition_conv_layer(child, rank, args.verbose))
            elif args.tt:
                tensor_train_decomposition_conv_layer(child, args.verbose)
            else:
                setattr(parent,name,tucker_decomposition_conv_layer(child, args.verbose))

            # model._modules[key] = decomposed
        
        decompose(child,name,ancestors)
            

def trace_handler(prof):
   print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))

def pthflops(model,input):
    flops = count_ops(model,input,verbose=False,print_readable=False)
    # multiply by 2 to convert from MAC to FLOP
    return 2*flops[0]
    

if __name__ == '__main__':
    args = get_args()
    tl.set_backend('pytorch')
    if args.train:
        # model = ModifiedVGG16Model().cuda()
        # model, input_size = initialize_model("alexnet",2,use_pretrained=True)
        model, input_size = initialize_model("resnet18",2,use_pretrained=True)
        model = model.cuda()
        print("model loaded")
        # optimizer = optim.SGD(model.classifier.parameters(), lr=0.0001, momentum=0.99)
        optimizer = optim.SGD(model.fc.parameters(), lr=0.0001, momentum=0.99)
        # optimizer = optim.SGD(model.classifier.parameters(), lr=0.0001, momentum=0.99)
        trainer = Trainer(args.train_path, args.test_path, model, optimizer, args.batch_size)

        trainer.train(epoches = 30)
        torch.save(model, "resnet18")

    elif args.decompose:
        model = torch.load("model").cuda()
        model.eval()
        model.cpu()
        
        # recurse over layers, decompose conv layers
        decompose(model,"root")
        print(model)

        # save model
        decomposed_name = "resnet18"
        if args.all_layers:
            decomposed_name += "_all"
        if args.cp :
            decomposed_name += "_cp"
        else:
            decomposed_name += "_tucker"
        
        torch.save(model, decomposed_name)

        


    elif args.fine_tune:
        model = torch.load(args.snap)

        for param in model.parameters():
            param.requires_grad = True

        print(model)
        model.cuda()        

        if args.cp:
            # optimizer = optim.SGD(model.parameters(), lr=0.000001)
            # optimizer = optim.Adam(model.parameters(),lr=0.00000001)
            optimizer = optim.Adam(model.parameters(),lr=0.001)

        else:
            # optimizer = optim.SGD(chain(model.features.parameters(), \
            #     model.classifier.parameters()), lr=0.01)
            optimizer = optim.SGD(model.parameters(), lr=0.001)
            # optimizer = optim.Adam(model.parameters(),lr=0.001)


        trainer = Trainer(args.train_path, args.test_path, model, optimizer, args.batch_size)

        trainer.test()
        model.cuda()
        model.train()
        trainer.train(epoches=100)
        model.eval()
        trainer.test()

        name = args.snap + "_tuned"

    elif args.profile:
        # profile baseline model

        name = str(args.batch_size)
        if args.snap == "":
            model, input_size = initialize_model("resnet18",2,use_pretrained=True)
            model.cuda()
            name = name + "_baseline"
        # profile snapshot (compressed model)
        else:
            model = torch.load(args.snap).cuda()
            name = name + "_" + args.snap
        
        
        
        input = torch.randn(args.batch_size,3,224,224).cuda()
        
        if args.fw:
            worker_name = "fw_" + name+"_"+str(time.time())
            model.eval()
        else:
            criterion = torch.nn.CrossEntropyLoss()
            target = torch.empty(args.batch_size, dtype=torch.long).random_(2).cuda()
            worker_name = name+"_"+str(time.time())
            model.train()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=10,
                active=10),
            # used when outputting for tensorboard
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                dir_name=os.path.join(".","profiles",name),
                worker_name=worker_name
            ),
            # on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_flops=True
    
            ) as p:
                if args.fw:
                    with torch.no_grad():
                        for iter in range(20):
                            print("Profiling step %s of 20"%str(iter+1))
                            output = model(input)
                            # send a signal to the profiler that the next iteration has started
                            p.step()
                else:
                     for iter in range(20):
                            print("Profiling step %s of 20"%str(iter+1))
                            output = model(input)
                            loss = criterion(output,target)
                            loss.backward()
                            # send a signal to the profiler that the next iteration has started
                            p.step()
        
        
        # computer flops
        print(pthflops(model,input))
    
    elif args.flops:
        name = str(args.batch_size)
        if args.snap == "":
            model, input_size = initialize_model("resnet18",2,use_pretrained=True)
            model.cuda()
            name = name + "_baseline"
        # profile snapshot (compressed model)
        else:
            model = torch.load(args.snap).cuda()
            name = name + "_" + args.snap
        
        
        
        input = torch.randn(args.batch_size,3,224,224).cuda()
        criterion = torch.nn.CrossEntropyLoss()
        target = torch.empty(args.batch_size, dtype=torch.long).random_(2).cuda()

         # computer flops
        print(pthflops(model,input))