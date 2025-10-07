"""
Script kiểm tra cấu hình hyperparameters từ checkpoint models.
"""
import torch

def check_model_config(model_path):
    """Kiểm tra cấu hình model từ checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print(f"\n{'='*80}")
    print(f"Model: {model_path.split('/')[-1]}")
    print(f"{'='*80}")
    
    # In ra các keys có trong checkpoint
    print("Keys trong checkpoint:", list(checkpoint.keys()))
    
    # Kiểm tra meta data
    if 'meta' in checkpoint:
        print("\n✓ Tìm thấy meta data:")
        meta = checkpoint['meta']
        if isinstance(meta, dict):
            for key, value in meta.items():
                print(f"  - {key}: {value}")
    
    # Kiểm tra xem có lưu config không
    if 'model_config' in checkpoint:
        print("✓ Tìm thấy model_config:")
        for key, value in checkpoint['model_config'].items():
            print(f"  - {key}: {value}")
        return checkpoint['model_config']
    
    # Lấy state_dict từ checkpoint
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    config = {}
    
    # In ra một số keys để debug
    print("\nMột số keys trong state_dict:")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        print(f"  {key}")
    if 'ca_block.ca.conv1.weight' in state_dict:
        print("  ...")
        print("  ca_block.ca.conv1.weight")
    print()
    
    # Kiểm tra BoT block
    if 'bot_block.conv1.weight' in state_dict:
        conv1_shape = state_dict['bot_block.conv1.weight'].shape
        bottleneck_dim = conv1_shape[0]
        config['bottleneck_dim'] = bottleneck_dim
        print(f"✓ BoT block detected:")
        print(f"  - bottleneck_dim: {bottleneck_dim}")
    
    # Kiểm tra CA block
    if 'ca_block.ca.conv1.weight' in state_dict:
        conv1_shape = state_dict['ca_block.ca.conv1.weight'].shape
        reduced_channels = conv1_shape[0]  # Số channels sau khi reduce
        
        # Tìm số channels gốc từ các layer trước CA block
        in_channels = None
        
        # Với ResNet18 (layer4 output là 512 channels)
        if 'layer4.1.conv2.weight' in state_dict:
            in_channels = state_dict['layer4.1.conv2.weight'].shape[0]
        elif 'layer4.0.conv2.weight' in state_dict:
            in_channels = state_dict['layer4.0.conv2.weight'].shape[0]
        # Với EfficientNetV2 (backbone blocks cuối cùng)
        elif 'backbone.conv_head.weight' in state_dict:
            in_channels = state_dict['backbone.conv_head.weight'].shape[1]
        # Tìm trong các conv layers khác  
        else:
            for key in state_dict.keys():
                if 'conv' in key and 'weight' in key and 'ca_block' not in key:
                    shape = state_dict[key].shape
                    if len(shape) == 4 and shape[0] > reduced_channels * 2:
                        in_channels = shape[0]
        
        if in_channels:
            reduction = in_channels // reduced_channels
            config['ca_reduction'] = reduction
            print(f"✓ Model with CA block detected:")
            print(f"  - in_channels: {in_channels}")
            print(f"  - reduced_channels: {reduced_channels}")
            print(f"  - ca_reduction: {reduction}")
    
    # Kiểm tra BoT block cho các model khác
    if 'bot_block.mhla.num_heads' in state_dict:
        num_heads = state_dict['bot_block.mhla.num_heads']
        config['num_heads'] = num_heads
        print(f"  - num_heads: {num_heads}")
    
    return config

# Kiểm tra các models bị lỗi
models_to_check = [
    'models/train 2/ResNet18_BoTLinear_pretrained_best.pt',
    'models/train 2/ResNet18_Hybrid_best.pt',
    'models/train 2/EfficientNetV2_S_CA_best.pt',
]

configs = {}
for model_path in models_to_check:
    try:
        config = check_model_config(model_path)
        configs[model_path] = config
    except Exception as e:
        print(f"❌ Lỗi khi kiểm tra {model_path}: {e}")

print(f"\n{'='*80}")
print("TỔNG KẾT CẤU HÌNH")
print(f"{'='*80}")
for model_path, config in configs.items():
    print(f"\n{model_path.split('/')[-1]}:")
    for key, value in config.items():
        print(f"  {key}: {value}")
