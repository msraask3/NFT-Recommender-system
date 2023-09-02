import pandas as pd
import numpy as np
import yaml
import os
import wandb
import argparse
import random

from recbole.evaluator import metrics
from recbole.config import Config
from recbole.data.dataset import Dataset
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from recbole.utils import get_model, get_trainer
from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function

# define arg parser
parser = argparse.ArgumentParser(description='recbole baseline')
parser.add_argument('--model', type=str, default='BPR')
parser.add_argument('--dataset', type=str, default='bayc')
parser.add_argument('--config', type=str, default='general')
parser.add_argument('--seed', type=str, default=2023)
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# args -> variables
MODEL = args.model
DATASET = args.dataset
CONFIG = f'config/fixed_config_{args.config}.yaml'
SEED = int(args.seed)

# functions
def objective_function(config_dict=None, config_file_list=None):
    """
    The objective function used for hyperparameter tuning.
    It is used inside the "main_HPO" function.
    """
    config = Config(model=MODEL, dataset=DATASET, config_dict=config_dict, config_file_list=config_file_list)
    config['eval_args']['order'] = 'MY'
    config['eval_args']['split'] = {'MY':f'dataset/collections/{DATASET}/split_indices.pkl'}
    config['log_wandb'] = True
    config['epochs'] = 50

    init_seed(SEED, config['reproducibility']) # init random seed
    dataset = create_dataset(config) # dataset creating and filtering # convert atomic files -> Dataset
    train_data, valid_data, test_data = data_preparation(config, dataset) # dataset splitting # convert Dataset -> Dataloader
    model = get_model(config['model'])(config, train_data.dataset).to(config['device']) # model loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model) # trainer loading and initialization

    # 1. Training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    # 2. Testing
    test_result = trainer.evaluate(test_data) 
    # 3. Save results
    return {
        'model': config['model'],
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }

def main_HPO():
    """
    The main function for experiment with hyperparameter tuning.
    """
    np.random.seed(SEED)
    random.seed(SEED)

    # run
    hp = HyperTuning(objective_function=objective_function, algo="exhaustive", early_stop=50,
                     max_evals=100, params_file=f'hyper/{MODEL}.hyper', fixed_config_file_list=[CONFIG])
    hp.run()
    
    # Save results (1): all results
    hp.export_result(output_file=f'hyper_result/{MODEL}_{DATASET}_{SEED}.result') 
    
    # Save results (2): best parameters
    with open(f'hyper_result/{MODEL}_{DATASET}_{SEED}.best_params', 'w') as file: 
        documents = yaml.dump(hp.best_params, file)
    best_result = hp.params2result[hp.params2str(hp.best_params)] # print best result

    # Save results (3): test result
    with open(f'hyper_result/{MODEL}_{DATASET}_{SEED}.test_result', 'w') as file:
        documents = yaml.dump(best_result['test_result'], file)
    wandb.log(best_result['test_result']) # log test result to wandb


def main():
    """
    The main function for experiment without hyperparameter tuning.
    """
    np.random.seed(SEED)
    random.seed(SEED)

    config = Config(model=MODEL, dataset=DATASET, config_file_list=[CONFIG])
    config['eval_args']['order'] = 'MY'
    config['eval_args']['split'] = {'MY':f'dataset/collections/{DATASET}/split_indices.pkl'}
    config['log_wandb'] = True
    config['epochs'] = 50
    
    init_seed(SEED, config['reproducibility']) # init random seed
    dataset = create_dataset(config) # dataset creating and filtering # convert atomic files -> Dataset
    train_data, valid_data, test_data = data_preparation(config, dataset) # dataset splitting # convert Dataset -> Dataloader
    model = get_model(config['model'])(config, train_data.dataset).to(config['device']) # model loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model) # trainer loading and initialization
    
    # 1. Training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    # 2. Testing
    test_result = trainer.evaluate(test_data)
    # 3. Save results
    print(test_result)
    wandb.log(test_result) # log test result to wandb



if __name__ == '__main__':

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID" # use GPU #0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    wandb.init(project="YOUR_PROJECT", name=f'{args.model}_{args.dataset}_{args.seed}', entity="YOUR_ENTITY", config=state)
    wandb.config.update(args)
    main_HPO()
    wandb.finish()

