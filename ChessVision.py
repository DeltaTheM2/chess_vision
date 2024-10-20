import torch
from pathlib import Path
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from train import ChessResNeXt

class ChessVision:
    def __init__(self, model_path, board_image_path, piece_folder):
        # Load the model
        self.model = ChessResNeXt()
        self.model.eval()  # Set model to evaluation mode
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))  # Load checkpoint
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)  # Load model weights

        # Preprocessing transformation (depends on model input size)
        self.preprocess = transforms.Compose([
            transforms.Resize((1024, 1024)),  # Assuming the input size expected by the model is 1024x1024
            transforms.ToTensor(),  # Convert frame to tensor
            transforms.Normalize(mean=[0.47225544, 0.51124555, 0.55296206],
                                 std=[0.27787283, 0.27054584, 0.27802786])
        ])

        # Load the chessboard image (for overlaying detected pieces)
        self.board_image = cv2.imread(board_image_path)
        if self.board_image is None:
            print(f"Error: Could not load board image from {board_image_path}")
        self.piece_folder = piece_folder

    def process_frame(self, frame):
        # Convert the frame (OpenCV format) to a PIL image for processing
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Preprocess the image
        input_tensor = self.preprocess(image).unsqueeze(0)  # Add batch dimension

        # Forward pass through the model to get predictions
        with torch.no_grad():
            logits = self.model(input_tensor)

        # Reshape the logits to a 64x13 tensor (64 squares, 13 possible classes)
        logits = logits.reshape((-1, 64, 13))

        # Post-process the logits to get the final prediction
        predicted_classes = torch.argmax(logits, dim=2).squeeze()

        # Convert predicted classes to human-readable form (e.g., map index to chess piece)
        piece_positions, piece_categories = self.decode_predictions(predicted_classes)

        # Overlay the 2D chessboard with detected pieces
        return self.create2D(frame, piece_positions, piece_categories)

    def decode_predictions(self, predicted_classes):
        """Map predicted classes to piece types and positions."""
        positions = []
        categories = []

        for i in range(64):
            class_id = predicted_classes[i].item()
            print(f"Square {i}: class_id = {class_id}")  # Debugging

            if class_id == 0:  # 'empty'
                continue

            if class_id >= 13 or class_id < 0:
                print(f"Error: Invalid class_id {class_id} for square {i}, skipping.")
                continue

            row, col = divmod(i, 8)
            positions.append((col, row))
            categories.append(self.map_class_to_piece(class_id))

        return positions, categories

    def map_class_to_piece(self, class_id):
        """Map class ID to piece name (customize this based on your class definitions)."""
        pieces = [
            'empty',           # class_id = 0
            'white-pawn',      # class_id = 1
            'white-knight',    # class_id = 2
            'white-bishop',    # class_id = 3
            'white-rook',      # class_id = 4
            'white-queen',     # class_id = 5
            'white-king',      # class_id = 6
            'black-pawn',      # class_id = 7
            'black-knight',    # class_id = 8
            'black-bishop',    # class_id = 9
            'black-rook',      # class_id = 10
            'black-queen',     # class_id = 11
            'black-king'       # class_id = 12
        ]

        if class_id >= len(pieces) or class_id < 0:
            print(f"Error: Invalid class_id {class_id}, setting to 'empty'")
            return 'empty'

        return pieces[class_id]

    def create2D(self, frame, positions, categories):
        """Overlay detected pieces on a 2D chessboard image."""
        piece_size = frame.shape[1] // 8  # Assuming the board is 8x8

        for piece, pos in zip(categories, positions):
            piece_path = Path(self.piece_folder) / f"{piece}.png"
            print(f"Loading piece image from: {piece_path}")

            # Check if the image file exists
            if not piece_path.exists():
                print(f"Error: Missing image for piece {piece}")
                continue

            piece_img = cv2.imread(str(piece_path), cv2.IMREAD_UNCHANGED)
            print(f"Loaded piece_img shape for {piece}: {piece_img.shape if piece_img is not None else 'None'}")

            # Ensure that the image was successfully loaded and has valid dimensions
            if piece_img is None or piece_img.size == 0:
                print(f"Error: Invalid image for piece {piece_path}")
                continue

            # Resize the piece image
            try:
                piece_img = cv2.resize(piece_img, (piece_size, piece_size))
                print(f"Resized piece_img shape for {piece}: {piece_img.shape}")
            except Exception as e:
                print(f"Error resizing image {piece_path}: {e}")
                continue

            # Check if the image has an alpha channel (for transparency)
            if piece_img.shape[2] == 4:  # Image has an alpha channel
                # Calculate position on the chessboard
                j, i = pos
                x, y = j * piece_size, i * piece_size

                # Ensure the offsets are within frame boundaries
                if y + piece_size > frame.shape[0] or x + piece_size > frame.shape[1]:
                    print(f"Error: Piece {piece} at position {pos} exceeds frame boundaries.")
                    continue

                # Overlay the piece image onto the board
                y_offset = y
                x_offset = x
                alpha = piece_img[:, :, 3] / 255.0

                # Avoid zero-sized regions
                if piece_img.shape[0] == 0 or piece_img.shape[1] == 0:
                    print(f"Error: Piece image {piece_path} has zero size after resizing.")
                    continue

                for c in range(3):  # Only process RGB channels, skip the alpha channel
                    try:
                        frame[y_offset:y_offset+piece_size, x_offset:x_offset+piece_size, c] = \
                            piece_img[:, :, c] * alpha + \
                            frame[y_offset:y_offset+piece_size, x_offset:x_offset+piece_size, c] * (1.0 - alpha)
                    except Exception as e:
                        print(f"Error overlaying piece {piece}: {e}")
            else:
                print(f"Warning: Piece image {piece_path} has no alpha channel, skipping transparency.")

        return frame

    def run(self):
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame for chess detection
            detected_frame = self.process_frame(frame)

            # Display the resulting frame
            cv2.imshow('Chess Vision', detected_frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and close windows
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "C:\\Users\\smirz\\OneDrive\\Documents\\Coding Minds\\Chess vision\\chess-vision\\checkpoint.ckpt"
    board_image_path = "C:\\Users\\smirz\\OneDrive\\Documents\\Coding Minds\\Chess vision\\end-to-end-chess-recognition\\resources\\board.png"
    piece_folder = "C:\\Users\\smirz\\OneDrive\\Documents\\Coding Minds\\Chess vision\\end-to-end-chess-recognition\\resources\\pieces\\"

    chess_vision = ChessVision(model_path, board_image_path, piece_folder)
    chess_vision.run()
