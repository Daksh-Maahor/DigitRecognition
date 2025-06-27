# Digit Recognition Neural Network

A Python-based digit recognition system using neural networks to classify handwritten digits from the MNIST dataset. This project includes both training and inference capabilities, with an interactive drawing interface for real-time digit recognition.

## Features

- **Neural Network Training**: Train a multi-layer perceptron on the MNIST dataset
- **Real-time Recognition**: Interactive drawing interface using Pygame
- **Data Augmentation**: Training with rotated images for better generalization
- **Multiple Training Scripts**: Various training configurations and network architectures
- **Visualization Tools**: View neural network data and predictions
- **Testing Framework**: Comprehensive testing scripts for model evaluation

## Project Structure

```
DigitRecognition/
├── data/
│   └── mnist.npz                 # MNIST dataset
├── main.py                       # Main training script
├── train.py                      # Training with data augmentation
├── train2.py - train5.py         # Alternative training configurations
├── program_real.py               # Interactive drawing interface
├── program.py - program3.py      # Alternative drawing interfaces
├── test.py - test5.py            # Testing scripts
├── view_neural_data.py - view_neural_data5.py  # Neural network visualization
├── data.py                       # Data loading utilities
├── get_wrong_predictions3.py     # Error analysis
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd DigitRecognition
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the MNIST dataset**:
   The project expects the MNIST dataset to be in `data/mnist.npz`. If you don't have it, you can download it or the project will handle it automatically.

## Usage

### Training the Model

**Basic Training** (main.py):
```bash
python main.py
```
This script trains a simple neural network with 100 epochs and displays accuracy after each epoch.

**Training with Data Augmentation** (train.py):
```bash
python train.py
```
This version includes random rotation augmentation for better model robustness.

**Alternative Training Configurations**:
```bash
python train2.py  # Different network architecture
python train3.py  # Alternative training parameters
python train4.py  # Extended training
python train5.py  # Advanced configuration
```

### Interactive Drawing Interface

**Real-time Digit Recognition**:
```bash
python program_real.py
```

**Controls**:
- **Mouse**: Draw digits on the canvas
- **C**: Clear the canvas
- **P**: Show the processed image plot
- **Close Window**: Exit the application

### Testing and Evaluation

**Basic Testing**:
```bash
python test.py
```

**Interactive Testing**:
```bash
python test2.py  # Interactive testing with keyboard controls
python test3.py  # Alternative testing interface
python test4.py  # Extended testing features
python test5.py  # Advanced testing
```

**Error Analysis**:
```bash
python get_wrong_predictions3.py
```

### Visualization

**View Neural Network Data**:
```bash
python view_neural_data.py
```

**Alternative Visualization Scripts**:
```bash
python view_neural_data2.py - view_neural_data5.py
```

## Neural Network Architecture

The project implements a feedforward neural network with the following characteristics:

- **Input Layer**: 784 neurons (28x28 pixel images)
- **Hidden Layer(s)**: Configurable (typically 128-256 neurons)
- **Output Layer**: 10 neurons (digits 0-9)
- **Activation Function**: Sigmoid
- **Learning Algorithm**: Backpropagation with gradient descent

## Data Processing

- **Input**: 28x28 grayscale images from MNIST dataset
- **Preprocessing**: Normalization to [0,1] range
- **Augmentation**: Random rotation (-60° to +60°) for training
- **Output**: One-hot encoded labels for digits 0-9

## Dependencies

- **numpy**: Numerical computations
- **matplotlib**: Plotting and visualization
- **pygame**: Interactive drawing interface
- **Pillow**: Image processing and augmentation
- **keyboard**: Keyboard input handling

## Performance

The trained model typically achieves:
- **Training Accuracy**: 85-95% (depending on configuration)
- **Real-time Recognition**: <100ms inference time
- **Robustness**: Good performance on rotated and slightly distorted digits

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- MNIST dataset creators
- Pygame community for the drawing interface
- NumPy and Matplotlib for scientific computing tools

## Troubleshooting

**Common Issues**:

1. **Pygame not working**: Ensure you have proper display setup for GUI applications
2. **Memory issues**: Reduce batch size or number of epochs in training scripts
3. **Slow performance**: Consider using GPU acceleration or reducing network size

**Getting Help**:
- Check the error messages in the console
- Verify all dependencies are installed correctly
- Ensure the MNIST dataset is in the correct location
