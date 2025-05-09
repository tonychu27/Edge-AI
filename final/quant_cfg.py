from hqq.core.quantize import BaseQuantizeConfig

# TODO: Make your own quant config for Language Model
def get_quant_config_slm(model):
    quant_config = {}
    
    n_layers = model.config.num_hidden_layers
    q2_config = BaseQuantizeConfig(nbits=4, group_size=64) 
    
    for i in range(n_layers):
        quant_config[f'model.layers.{i}.self_attn.q_proj'] = BaseQuantizeConfig(nbits=4, group_size=32) 
        quant_config[f'model.layers.{i}.self_attn.k_proj'] = BaseQuantizeConfig(nbits=4, group_size=32) 
        quant_config[f'model.layers.{i}.self_attn.v_proj'] = BaseQuantizeConfig(nbits=4, group_size=32) 
        quant_config[f'model.layers.{i}.self_attn.o_proj'] = BaseQuantizeConfig(nbits=4, group_size=32) 
        
        quant_config[f'model.layers.{i}.mlp.gate_proj'] = BaseQuantizeConfig(nbits=4, group_size=32) 
        quant_config[f'model.layers.{i}.mlp.up_proj'] = BaseQuantizeConfig(nbits=4, group_size=32) 
        quant_config[f'model.layers.{i}.mlp.down_proj'] = BaseQuantizeConfig(nbits=8, group_size=64) 
    
    # q2_config = BaseQuantizeConfig(nbits=4, group_size=64) 
    
    # for i in range(n_layers):
    #     quant_config[f'model.layers.{i}.self_attn.q_proj'] = q2_config 
    #     quant_config[f'model.layers.{i}.self_attn.k_proj'] = q2_config 
    #     quant_config[f'model.layers.{i}.self_attn.v_proj'] = q2_config 
    #     quant_config[f'model.layers.{i}.self_attn.o_proj'] = q2_config 
        
    #     quant_config[f'model.layers.{i}.mlp.gate_proj'] = q2_config 
    #     quant_config[f'model.layers.{i}.mlp.up_proj'] = q2_config 
    #     quant_config[f'model.layers.{i}.mlp.down_proj'] = q2_config 
        
    return quant_config