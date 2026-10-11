"""Recompute the printed scalar examples using arithmetic and actual tensors."""
import math
import sys
import torch
from .common import ROOT, write_json
from .checks import numerical_examples


def main():
    sys.path.insert(0, str(ROOT / 'mnist_web'))
    from model import MnistCNN
    values = {
        'distance_120_022': sum((a-b)**2 for a,b in zip([1,2,0],[0,2,2])),
        'dot_123_20m1': sum(a*b for a,b in zip([1,2,3],[2,0,-1])),
        'population_variance_012': sum((a-1)**2 for a in [0,1,2])/3,
        'population_std_012': math.sqrt(2/3),
        'normalized_black': -0.1307/0.3081,
        'normalized_white': (1-0.1307)/0.3081,
        'weighted_loss_64_6': (64*1+6*3)/70,
        'weighted_accuracy_100_10': (90+5)/110,
        'ce_examples': {str(p):-math.log(p) for p in [.9,.5,.1,.25]},
        'sgd_weight_only_loss': (2*.9-3)**2/2,
        'sgd_weight_bias_loss': (2*.9+.2-3)**2/2,
        'sgd_large_step_loss': (2*4.5+2-3)**2/2,
        'project_parameters': sum(p.numel() for p in MnistCNN().parameters()),
        'fc1_parameter_fraction': 401536/421834,
        'full_batches': math.ceil(55000/64),
        'last_batch_size': 55000%64,
        'confusion_accuracy': 23/30,
        'class2_precision': 9/12, 'class2_recall': 9/10,
        'class2_f1': 18/22,
        'adam_first_m': .1*2/(1-.9),
        'adam_first_v': .001*4/(1-.999),
    }
    x=torch.tensor([[[[1.,2.,0.],[0.,1.,3.],[2.,0.,1.]]]])
    k=torch.tensor([[[[1.,0.],[0.,-1.]]]])
    conv=torch.nn.functional.conv2d(x,k)
    assert conv.tolist()==[[[[0.,-1.],[0.,0.]]]]
    values['sliding_kernel']=conv.tolist()
    # Wilson interval assumes independent representative Bernoulli sampling.
    n,p,z=10,.8,1.959963984540054
    denominator=1+z*z/n
    center=(p+z*z/(2*n))/denominator
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    values['wilson_8_of_10_95']=[center-half,center+half]
    r,j,rf=1,1,[]
    for kernel,stride in [(3,1),(2,2),(3,1),(2,2)]:
        r += (kernel-1)*j
        j *= stride
        rf.append([r,j])
    values['project_receptive_field']=rf
    assert values['project_parameters']==421834
    assert values['full_batches']==860 and values['last_batch_size']==24
    assert rf==[[3,1],[4,2],[8,2],[10,4]]
    assert abs(values['adam_first_m']-2)<1e-12
    assert abs(values['adam_first_v']-4)<1e-12
    values['tiny_backprop_and_teaching_parameters']=numerical_examples()
    write_json(ROOT/'textbook/evidence/numerical_audit.json',values)
    print('Numerical examples recomputed and invariants checked.')


if __name__=='__main__':main()
