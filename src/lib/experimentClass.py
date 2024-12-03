import os, sys
import timm
from deepinv.optim.optimizers import optim_builder
from deepinv.physics.generator import PhysicsGenerator
from deepinv.physics.forward import LinearPhysics
from typing import Callable
import torch
import re
import pandas as pd
from wildlife_tools.data.dataset import WildlifeDataset



class ExpSetup:
    def __init__(
        self,
        work_dir = None,
        src_dir = None,
        data_dir = None,
        model_dir=None,
        out_dir = None,
        datasets_dir = None,
    ):

        self.work_dir   = work_dir if work_dir != None else os.getcwd()
        self.src_dir    = src_dir if src_dir != None else self.work_dir + '/src'
        self.data_dir   = data_dir if data_dir != None else self.work_dir + '/data'
        self.datasets_dir   = datasets_dir if datasets_dir != None else self.work_dir + '/datasets'
        self.model_dir  = model_dir if model_dir != None else self.work_dir + '/models'
        self.out_dir    = out_dir if out_dir != None else self.work_dir + '/output'
        self.createDir()
        self.optimizers={}
    
    def createDir(
            self,
            dir='all'
    ):
        if dir == 'all':
            os.makedirs(self.work_dir, exist_ok=True)
            os.makedirs(self.src_dir, exist_ok=True)
            os.makedirs(self.data_dir, exist_ok=True)
            os.makedirs(self.datasets_dir, exist_ok=True)
            os.makedirs(self.model_dir, exist_ok=True)
            os.makedirs(self.out_dir, exist_ok=True)
        elif dir == 'work':
            os.makedirs(self.work_dir, exist_ok=True)        
        elif dir == 'src':
            os.makedirs(self.src_dir, exist_ok=True)
        elif dir == 'data':
            os.makedirs(self.data_dir, exist_ok=True)
        elif dir == 'datasets':
            os.makedirs(self.datasets_dir, exist_ok=True)
        elif dir == 'model':
            os.makedirs(self.model_dir, exist_ok=True)
        elif dir == 'output':
            os.makedirs(self.out_dir, exist_ok=True)

    def set_workDir(
        self,
        work_dir
    ):
        
        self.work_dir   = work_dir
        self.src_dir    = self.work_dir + '/src'
        self.data_dir   = self.work_dir + '/data'
        self.model_dir  =  self.work_dir + '/models'
        self.out_dir    = self.work_dir + '/output'
        self.createDir()

    def set_srcDir(
        self,
        src_dir
    ):
        self.src_dir = src_dir
        self.createDir(dir='src')


    def set_dataDir(
        self,
        data_dir
    ):
        self.data_dir = data_dir
        self.createDir(dir='data')

    def set_datasetsDir(
        self,
        datasets_dir
    ):
        self.datasets_dir = datasets_dir
        self.createDir(dir='datasets')

    
    def set_modelDir(
        self,
        model_dir
    ):
        self.model_dir = model_dir
        self.createDir(dir='model')

    def set_outDir(
        self, 
        out_dir
    ):
        self.out_dir = out_dir
        self.createDir(dir='output')

    def get_dataPath(
        self,
        data
    ):
        
        data = self.data_dir + '/' + data
        if not os.path.exists(data):
            print('The data path: ', data, ' does not exist.', file=sys.stderr)
            return
        if not os.path.isdir(data):
            print('The path: ', data, ' is not a directory.', file=sys.stderr)
            return
        self.data = data
        return self.data
    
    def create_model(
        self, 
        model_name, 
        pretrained = False, 
        **kwargs
    ):
        modified_model_name = model_name.split('/', 1)[1]
        model_path = self.model_dir+ '/' + modified_model_name + '.pt'
        if pretrained:
            model = timm.create_model(model_name=model_name, pretrained = True, **kwargs)
            torch.save(model.state_dict(), f=model_path)
            return model
        
        if not os.path.exists(model_path):
            print('The model path:', model_path, ' does not exist. Initializing with random weights.')
            return
            
        return timm.create_model(model_name=model_name,
                checkpoint_path=model_path, **kwargs)
        
    def get_optimizer(
        self,
        name  
    ):
        if name not in self.optimizers:
            print('The optimizer ', name, ' has not been created')
            return
        return self.optimizers[name]  
        
    
    def create_optimizer(
        self,
        name,
        iteration,
        max_iter=100,
        params_algo={"lambda": 1.0, "stepsize": 1.0, "g_param": 0.05},
        data_fidelity=None,
        prior=None,
        F_fn=None,
        g_first=False,
        overwrite=False,
        **kwargs,
    ):
        if name in self.optimizers and not overwrite:
            return self.optimizers[name]
        

        model = optim_builder(
                    iteration=iteration,
                    max_iter=max_iter,
                    params_algo=params_algo,
                    data_fidelity=data_fidelity,
                    prior=prior,
                    F_fn=F_fn,
                    g_first=g_first,
                    **kwargs
                    )
        self.optimizers[name] = model
        return model
    
    def get_endswth_paths(
        self,
        root_folder,
        endswith='.h5',
        sort=False
        ):
            h5_paths = []
            for root, dirs, files in os.walk(root_folder):
                for file in files:
                    if file.endswith(endswith):
                        h5_paths.append(os.path.join(root, file))

            if sort:
                h5_paths = sorted(h5_paths, key=lambda s: [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]) 
            return h5_paths


class CombinedCallable:
    def __init__(
        self, 
        *functions : Callable,
        generator : PhysicsGenerator = None
        ):
        self.functions = functions
        self.generator = generator

    def __call__(
        self,
        x
        ):
        result = x
        for func in self.functions:
            if self.generator is not None and isinstance(func, LinearPhysics):
                params = self.generator.step(result.size(0))
                result = func(result, **params)
            else:
                result = func(result)
        return result
    

class CombinedClass:
    def __init__(
            self,
            classA,
            classB,
        ):

            if len(classA) != len(classB):
                raise ValueError("Both classes must have the same length.")
            
            self.classA = classA
            self.classB = classB
    
    def __getitem__(self, index):
        itemA = self.classA[index]
        if isinstance(itemA, tuple):
            itemA = itemA[0]

        itemB = self.classB[index]
        if isinstance(itemB, tuple):
            itemB = itemB[0]

        return itemA, itemB
     
        

    def __len__(self):
        return len(self.classA)


def get_all_metadata(path):
    metadata = {}
    data_gt = pd.read_csv(os.path.join(path, "Ground truth.csv"))
    data_nl = pd.read_csv(os.path.join(path, "No learning.csv"))
    data_rc = pd.read_csv(os.path.join(path, "Reconstruction.csv"))
    
    metadata["gt"] = data_gt
    metadata["nl"] = data_nl
    metadata["rc"] = data_rc

    return metadata