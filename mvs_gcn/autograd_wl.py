import torch

"""
Only use the last layer
"""
def capture_activations(layer, inputs, outputs):
    setattr(layer, "activations", inputs[0].detach())
    
def capture_backprops(layer, inputs, outputs):
    setattr(layer, "backprops", outputs[0].detach())
    
def calculate_sample_grad(layer):
    A = layer.activations
    B = layer.backprops
    
    n = A.shape[0]
    B = B * n
    weight_grad = torch.einsum('ni,nj->nij', B, A)
    bias_grad = B
    grad_norm = torch.sqrt(weight_grad.norm(p=2, dim=(1,2)).pow(2) + bias_grad.norm(p=2, dim=1).pow(2)).squeeze().detach()
    return grad_norm

"""
Use all layers
"""
def capture_activations_exact(layer, inputs, outputs):
    setattr(layer, "activations", inputs[0].detach())
    
def capture_backprops_exact(layer, inputs, outputs):
    if not hasattr(layer, 'backprops_list'):
        setattr(layer, 'backprops_list', [])
    layer.backprops_list.append(outputs[0].detach())

def add_hooks(model):
    for layer in model.modules():
        if layer.__class__.__name__=='Linear':
            layer.register_forward_hook(capture_activations_exact)
            layer.register_backward_hook(capture_backprops_exact)

def calculate_exact_sample_grad(model):
    grad_norm_sum = None
    for layer in model.modules():
        if layer.__class__.__name__!='Linear':
            continue
        A = layer.activations
        n = A.shape[0]
        
        B = layer.backprops_list[0]
        B = B*n
        
        weight_grad = torch.einsum('ni,nj->nij', B, A)
        bias_grad = B
        grad_norm = weight_grad.norm(p=2, dim=(1,2)).pow(2) + bias_grad.norm(p=2, dim=1).pow(2)
        if grad_norm_sum is None:
            grad_norm_sum = grad_norm
        else:
            grad_norm_sum += grad_norm
    grad_norm = torch.sqrt(grad_norm_sum).squeeze().detach()
    return grad_norm

def del_backprops(model):
    for layer in model.modules():
        if hasattr(layer, 'backprops_list'):
            del layer.backprops_list