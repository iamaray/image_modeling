import torch
from data.process_ciphar_10 import process_cifar10
import argparse

from dsno.model import DSNO
from dsno.trainer import train
from evaluation import evaluate_model, plot_training_curves

def process():
    process_cifar10(batch_size=16)
    
def train_model(epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DSNO()
    train_losses = train(
        teacher_data_path='data/cifar10_teacher.pt',
        model=model,
        batch_size=32,
        epochs=epochs,
        device=device,
        save_every=20
    )
    return train_losses

def evaluate(model_path='checkpoints/dsno_final.pth', num_samples=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = evaluate_model(model_path, device, num_samples)
    return results

def main():
    parser = argparse.ArgumentParser(description='DSNO Training Pipeline')
    parser.add_argument('command', choices=['process', 'train', 'evaluate', 'plot'], 
                       help='Command to run: process, train, evaluate, or plot')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--model_path', type=str, default='checkpoints/dsno_final.pth',
                       help='Path to model for evaluation')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples for evaluation')
    
    args = parser.parse_args()
    
    if args.command == 'process':
        print("Processing CIFAR-10 data...")
        process()
        print("Data processing complete!")
        
    elif args.command == 'train':
        print(f"Training model for {args.epochs} epochs...")
        train_losses = train_model(epochs=args.epochs)
        print("Training complete!")
        print(f"Final loss: {train_losses[-1]:.6f}")
        
    elif args.command == 'evaluate':
        print("Evaluating model...")
        results = evaluate(args.model_path, args.num_samples)
        print("Evaluation complete!")
        
    elif args.command == 'plot':
        print("Plotting training curves...")
        plot_training_curves()

if __name__ == "__main__":
    main()