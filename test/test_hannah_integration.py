# This is a test for the hannah/tvm integration

from pytest import importorskip

hannah = importorskip("hannah")
tvm = importorskip("tvm")

import omegaconf    

from hydra import compose, initialize_config_module 

from hannah.nas.functional_operators.op import Tensor, FloatType
from hannah.models.conv_vit.models  import conv_vit

from hannah_tvm import config
from hannah_tvm.backend import TVMBackend

from lightning.pytorch import LightningModule

import torch
import torch.nn as nn


class SimpleModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(5, 5)

        self.example_input_array = torch.tensor([1.0]*5).unsqueeze(0)
        self.example_feature_array = torch.tensor([1.0]*5).unsqueeze(0)   
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
    
    def prepare_data(self):
        pass
    
    def get_class_names(self):
        return ["class1", "class2", "class3", "class4", "class5"] 


def init_backend():
    # initialize hydra
    with initialize_config_module(config_module="hannah_tvm.conf.backend", job_name="test_hannah_integration", version_base="1.2"):
        cfg : omegaconf.DictConfig = compose(config_name="tvm", overrides=["tuner=meta_scheduler", "board=local_cpu"])
        
        print(omegaconf.OmegaConf.to_yaml(cfg)) 
        
        backend = TVMBackend(cfg.board, cfg.tuner)
        
        return backend

def test_simple():
    module = SimpleModule() 
    module.prepare_data()
    module.setup("fit")
    
    x = torch.tensor([1.0]*5).unsqueeze(0)
    backend = init_backend()
    
    backend.prepare(module)


#def test_conv_vit():    
#    input = Tensor(shape=(1, 3, 224, 224), name="input", dtype=FloatType(), axis = ["N", "C", "H", "W"])
#    model = conv_vit("Vision_transformer", input)
#    
#    backend = init_backend()
        
        
        
     
