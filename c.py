import time
import cv2
import numpy as np
from picamera2 import Picamera2
import tensorflow as tf
import chess
import chess.engine

# Initialize the camera
picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (1280, 720)})
picam2.configure(camera_config)
picam2.start()
time.sleep(1)  # Allow the camera to warm up

# Initialize chess board
board = chess.Board()

# Load the piece recognition model
model = tf.keras.models.load_model('models/chess_piece_classifier.h5')

# Labels for piece classification
labels = [
    'empty',
    'black_bishop', 'black_king', 'black_knight', 'black_pawn', 'black_queen', 'black_rook',
    'white_bishop', 'white_king', 'white_knight', 'white_pawn', 'white_queen', 'white_rook'
]

def capture_image():
    """Capture an image from the camera."""
    image = picam2.capture_array()
    return image

def preprocess_image(image):
    """Preprocess the captured image for analysis."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

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

def classify_pieces(squares):
    """Classify the piece on each square."""
    piece_positions = []
    for square in squares:
        # Preprocess the square image
        img = cv2.resize(square, (64, 64))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict the piece
        prediction = model.predict(img)
        label_idx = np.argmax(prediction, axis=1)[0]
        label = labels[label_idx]
        piece_positions.append(label)
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
    previous_squares = get_chessboard_grid(previous_image)
    previous_positions = classify_pieces(previous_squares)

    try:
        while True:
            # Capture the current board state
            current_image = capture_image()
            current_squares = get_chessboard_grid(current_image)
            current_positions = classify_pieces(current_squares)

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
