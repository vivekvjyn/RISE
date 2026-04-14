from torch import nn
from peft import LoraConfig, get_peft_model

def get_layers(model):
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            layers.append(name)
    return layers

def apply_lora(model, r=8, alpha=16, dropout=0.05):
    target_modules = get_layers(model)
    config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", target_modules=target_modules)
    model = get_peft_model(model, config)
    return model
