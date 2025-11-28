import torch
import sys

# 加载一个权重文件
weight_file = 'model/sac_v2_bus_ensemble/replay_buffer_size_1000000/critic_actor_ratio_2/maximum_alpha_0.3/weight_reg_0.03 0'

print(f"Loading weight file: {weight_file}")
print("=" * 80)

data = torch.load(weight_file, map_location='cpu')

print('Keys in the weight file:')
print(list(data.keys()))
print()

# 查看每个key的详细信息
for key in data.keys():
    print(f'{key}:')
    if isinstance(data[key], dict):
        print(f'  Type: dict with {len(data[key])} keys')
        print(f'  Keys: {list(data[key].keys())}')

        # 如果是state_dict，显示参数名和形状
        if 'state_dict' in key or key in ['actor_state_dict', 'critic_state_dict',
                                           'critic_target_state_dict', 'log_alpha']:
            print(f'  Parameters:')
            for param_name, param_value in list(data[key].items())[:5]:
                if hasattr(param_value, 'shape'):
                    print(f'    {param_name}: shape {param_value.shape}')
                else:
                    print(f'    {param_name}: {param_value}')
            if len(data[key]) > 5:
                print(f'    ... and {len(data[key]) - 5} more parameters')
    else:
        print(f'  Type: {type(data[key])}')
        if hasattr(data[key], 'shape'):
            print(f'  Shape: {data[key].shape}')
        else:
            print(f'  Value: {data[key]}')
    print()

print("=" * 80)
print("\nConclusion:")
print("This weight file contains:")
for key in data.keys():
    if 'actor' in key.lower():
        print(f"  - {key}: Actor network parameters")
    elif 'critic' in key.lower():
        print(f"  - {key}: Critic network parameters")
    elif 'alpha' in key.lower():
        print(f"  - {key}: Temperature parameter for SAC")
    elif 'optimizer' in key.lower():
        print(f"  - {key}: Optimizer state")
    else:
        print(f"  - {key}: {type(data[key])}")
