import torch
import random
import os
import numpy as np

from hqq.utils.patching import recommended_inductor_config_setter
from hqq_utils import AutoHQQTimmModel, get_size_of_model
from utils import prepare_data, evaluate_model

from quant_cfg import get_quant_config_deit

def main():
    ############## Set Up ##############
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    recommended_inductor_config_setter()
    
    device = 'cuda:0'
    batch_size = 16
    _, test_loader, _ = prepare_data(batch_size)
    
    model = torch.load('./0.9099_deit3_small_patch16_224.pth', map_location='cpu', weights_only=False)
    model = model.to(device)
    model.eval()
    
    # Config to align HQQ 
    model.device = 'cuda:0'
    model.dtype = torch.float32
    ##################################### 

    # TODO: Quantize
    quant_config = get_quant_config_deit(model)
    
    AutoHQQTimmModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.float32, device=device)
    
    from hqq.utils.patching import prepare_for_inference
    prepare_for_inference(model)
    torch.cuda.empty_cache()
    
    acc_after_quant = evaluate_model(model, test_loader, 'cuda:0')
    print(f'Accuracy After Quant: {acc_after_quant}%')
    print(f'Model Size (MiB) {get_size_of_model(model)/ (1024 ** 2)} MiB')
    
    score = 20 - max(0, 90 - acc_after_quant) * 10 + (17 - get_size_of_model(model) / (1024 ** 2))
    print(f'Score: {score}')

    
if __name__ == '__main__':
    main()