import time
import cv2
import numpy as np
from picamera2 import Picamera2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import chess

# Initialize the camera
picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (1024, 1024)})
picam2.configure(camera_config)
picam2.start()
time.sleep(1)  # Allow the camera to warm up

# Initialize chess board
board = chess.Board()

# Define the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture
class ChessResNeXt(nn.Module):
    """Modified ResNeXt network for chess recognition."""

    def __init__(self):
        super(ChessResNeXt, self).__init__()

        backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnext101_32x8d', pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]

        self.feature_extractor = nn.Sequential(*layers)

        num_target_classes = 64 * 13  # 64 squares * 13 classes

        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        x = self.feature_extractor(x).flatten(1)
        x = self.classifier(x)
        return x

# Instantiate the model and load the checkpoint
model = ChessResNeXt().to(device)
checkpoint_path = 'path_to_your_checkpoint.ckpt'  # Update with your checkpoint path
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()  # Set model to evaluation mode

# Labels for piece classification
labels = [
    'empty',
    'black_bishop', 'black_king', 'black_knight', 'black_pawn', 'black_queen', 'black_rook',
    'white_bishop', 'white_king', 'white_knight', 'white_pawn', 'white_queen', 'white_rook'
]

# Define the image transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(1024),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.47225544, 0.51124555, 0.55296206],
        std=[0.27787283, 0.27054584, 0.27802786]),
])

def capture_image():
    """Capture an image from the camera."""
    image = picam2.capture_array()
    return image

def get_chessboard_grid(image):
    """Detect the chessboard grid and extract individual squares."""
    # Assume the board occupies the full image for simplicity
    # Adjust based on your setup

    # Image dimensions
    height, width = image.shape[:2]
    square_size_y = height // 8
    square_size_x = width // 8

    squares = []
    for row in range(8):
        for col in range(8):
            x_start = col * square_size_x
            y_start = row * square_size_y
            square = image[y_start:y_start+square_size_y, x_start:x_start+square_size_x]
            squares.append(square)
    return squares

def classify_pieces(image):
    """Classify the pieces on the board."""
    # Apply the transformations
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        output = output.view(-1, 64, 13)  # Batch size x 64 squares x 13 classes
        predictions = torch.argmax(output, dim=2).cpu().numpy()[0]

    # Map predictions to labels
    piece_positions = [labels[pred] for pred in predictions]
    return piece_positions

def detect_move(previous_positions, current_positions):
    """Detect the move made by comparing board states."""
    move = {}
    for idx in range(64):
        prev_piece = previous_positions[idx]
        curr_piece = current_positions[idx]
        if prev_piece != curr_piece:
            if prev_piece != 'empty' and curr_piece == 'empty':
                move['from'] = idx
            elif prev_piece == 'empty' and curr_piece != 'empty':
                move['to'] = idx
            else:
                # Handle special cases like captures or promotions
                move['from'] = move.get('from', idx)
                move['to'] = move.get('to', idx)
    return move if 'from' in move and 'to' in move else None

def index_to_coordinates(index):
    """Convert a square index to board coordinates (e.g., 0 -> 'a8')."""
    row = index // 8
    col = index % 8
    files = 'abcdefgh'
    ranks = '87654321'
    return files[col] + ranks[row]

def update_board(board, move):
    """Update the chess board with the detected move."""
    from_square = index_to_coordinates(move['from'])
    to_square = index_to_coordinates(move['to'])
    uci_move = from_square + to_square
    chess_move = chess.Move.from_uci(uci_move)

    if chess_move in board.legal_moves:
        board.push(chess_move)
        print(f"Move made: {from_square} to {to_square}")
        print(board)
    else:
        print(f"Illegal move detected: {from_square} to {to_square}")

def main():
    # Capture the initial board state
    previous_image = capture_image()
    previous_positions = classify_pieces(previous_image)

    try:
        while True:
            # Capture the current board state
            current_image = capture_image()
            current_positions = classify_pieces(current_image)

            # Detect move
            move = detect_move(previous_positions, current_positions)

            if move:
                # Update the board
                update_board(board, move)
                # Update previous positions
                previous_positions = current_positions.copy()

            # Wait before capturing the next image
            time.sleep(1)

    except KeyboardInterrupt:
        # Handle user interruption
        print("Program terminated by user.")

    finally:
        # Ensure the camera is stopped properly
        picam2.stop()

if __name__ == '__main__':
    main()
