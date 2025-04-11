from hqq.core.quantize import BaseQuantizeConfig

# TODO: Make your own quant config for DeiT-S
def get_quant_config_deit(model):
    quant_config = {}
    
    n_blocks = len(model.blocks)
    q2_config = BaseQuantizeConfig(nbits=4, group_size=64)
    q4_config = BaseQuantizeConfig(nbits=8, group_size=256)
    
    for i in range(n_blocks):
        quant_config[f'blocks.{i}.attn.qkv'] = q2_config
        quant_config[f'blocks.{i}.attn.proj'] = q2_config
        quant_config[f'blocks.{i}.mlp.fc1'] = q2_config
        quant_config[f'blocks.{i}.mlp.fc2'] = q2_config
    
    for i in [0, 1]:
        quant_config[f'blocks.{i}.attn.proj'] = q4_config
        quant_config[f'blocks.{i}.mlp.fc1'] = q4_config
        quant_config[f'blocks.{i}.mlp.fc2'] = q4_config
    
    return quant_config

# TODO: Make your own quant config for Language Model
def get_quant_config_slm(model):
    quant_config = {}
    
    n_layers = model.config.num_hidden_layers
    q_config_attn = BaseQuantizeConfig(nbits=8, group_size=64)
    q_config_mlp = BaseQuantizeConfig(nbits=4, group_size=128)

    for i in range(n_layers):
        # Attention projections
        quant_config[f'model.layers.{i}.self_attn.q_proj'] = q_config_attn
        quant_config[f'model.layers.{i}.self_attn.k_proj'] = q_config_attn
        quant_config[f'model.layers.{i}.self_attn.v_proj'] = q_config_attn
        quant_config[f'model.layers.{i}.self_attn.o_proj'] = q_config_attn

        # MLP projections
        quant_config[f'model.layers.{i}.mlp.gate_proj'] = q_config_mlp
        quant_config[f'model.layers.{i}.mlp.up_proj'] = q_config_mlp
        quant_config[f'model.layers.{i}.mlp.down_proj'] = q_config_mlp
        
    return quant_config