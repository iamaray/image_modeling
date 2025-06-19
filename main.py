import torch
from data.process_ciphar_10 import process_ciphar10
import matplotlib.pyplot as plt

from dsno.model import DSNO

def main():
    process_ciphar10(batch_size=64)
    
if __name__ == "__main__":
    # main()
    
    sample = torch.rand((64, 3, 32, 32))
    
    model = DSNO()
    out = model(sample, torch.Tensor([100, 200, 300, 500, 550]))
    print(out.shape)