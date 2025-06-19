import torch
from data.process_ciphar_10 import process_ciphar10

def main():
    process_ciphar10(batch_size=64)
    
if __name__ == "__main__":
    main()