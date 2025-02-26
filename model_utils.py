import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchcontrib.optim import SWA

from data_utils import (Dataset_ASVspoof2019_train, Dataset_ASVspoof2019_devNeval, genSpoof_list)
from evaluation import calculate_tDCF_EER
from model_utils import create_optimizer, seed_worker, set_seed, str_to_bool

from importlib import import_module

def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module(model_config["architecture"])
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


# new dataloader
def get_loader(
        database_path: str,
        protocol_path: str,
        seed: int,
        config: dict,
        data_type: str) -> List[torch.utils.data.DataLoader]:

    
    if data_type == 'eval':

        file_eval = genSpoof_list(dir_meta=protocol_path,
                                    is_train=False,
                                    is_eval=True)

        eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval,
                                                base_dir=database_path,
                                                audio_ext=config["data_config"]["audio_ext"])
        
        data_loader = DataLoader(eval_set,
                                 batch_size=config["train_config"]["batch_size"],
                                 shuffle=False,
                                 drop_last=False,
                                 pin_memory=True)
        
    elif data_type == 'dev':

        _, file_dev = genSpoof_list(dir_meta=protocol_path,
                                        is_train=False,
                                        is_eval=False)
        
        print("no. validation files:", len(file_dev))

        dev_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_dev,
                                                base_dir=database_path,
                                                audio_ext=config["data_config"]["audio_ext"])
        data_loader = DataLoader(dev_set,
                                batch_size=config["train_config"]["batch_size"],
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)
        
    elif data_type == 'train':
            
        d_label_trn, file_train = genSpoof_list(dir_meta=protocol_path,
                                                is_train=True,
                                                is_eval=False)

        print("no. training files:", len(file_train))

        train_set = Dataset_ASVspoof2019_train(list_IDs=file_train,
                                            labels=d_label_trn,
                                            base_dir=database_path,
                                            audio_ext=config["data_config"]["audio_ext"])
        gen = torch.Generator()
        gen.manual_seed(seed)
        data_loader = DataLoader(train_set,
                                batch_size=config["train_config"]["batch_size"],
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True,
                                worker_init_fn=seed_worker,
                                generator=gen)

    return data_loader
