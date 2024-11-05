"""This module implements the Browser class with real-time camera integration."""

import argparse
from pathlib import Path
from tkinter import ttk
import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageTk
from ttkthemes import ThemedTk

from picamera2 import Picamera2

# Import PyTorch and your model architecture
import torch
from torchvision import transforms
from torch import nn
from torchvision import models

# Window width for the Browser App
WINDOW_WIDTH = 1280


class ChessResNeXt(nn.Module):
    """Modified ResNeXt network for chess recognition on ChessReD."""

    def __init__(self):
        super().__init__()

        backbone = models.resnext101_32x8d(weights=None)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]

        self.feature_extractor = nn.Sequential(*layers)

        num_target_classes = 64 * 13

        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        x = self.feature_extractor(x).flatten(1)
        x = self.classifier(x)
        return x


class ChessPieceDetector:
    """Class to handle loading the model and making predictions."""

    def __init__(self, model_path):
        """Load the pre-trained model."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize the model architecture
        self.model = ChessResNeXt()
        self.model.to(self.device)

        # Load the pre-trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

        # Define the image transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(512),  # Reduced from 1024
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.47225544, 0.51124555, 0.55296206],
                std=[0.27787283, 0.27054584, 0.27802786]),
        ])

    def predict(self, image):
        """Make predictions on the input image."""
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(input_tensor)

        # Process outputs to get predictions
        positions, categories = self.process_outputs(logits)
        return positions, categories

    def process_outputs(self, logits):
        """Process model outputs to extract positions and categories."""
        # The logits are of shape (1, 64 * 13)
        # Reshape to (1, 64, 13)
        logits = logits.view(-1, 64, 13)
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=2)
        # Get predicted classes for each square
        preds = torch.argmax(probs, dim=2).squeeze(0).cpu().numpy()

        # Map class indices to piece names
        class_map = [
            'empty',
            'white_pawn', 'white_knight', 'white_bishop', 'white_rook', 'white_queen', 'white_king',
            'black_pawn', 'black_knight', 'black_bishop', 'black_rook', 'black_queen', 'black_king'
        ]

        positions = []
        categories = []

        cols = 'abcdefgh'
        rows = '87654321'  # Since the first row corresponds to row 8 in chess notation

        for idx, class_idx in enumerate(preds):
            if class_idx == 0:
                continue  # Skip empty squares
            piece = class_map[class_idx]
            col = cols[idx % 8]
            row = rows[idx // 8]
            position = f"{col}{row}"
            positions.append(position)
            categories.append(piece)

        return positions, categories


class Browser:
    """Browser class.

    The Browser class captures images from the Raspberry Pi camera,
    processes them using a pre-trained model to detect chess pieces,
    and visualizes the game in a Tkinter GUI.
    """

    def __init__(self, model_path: Path) -> None:
        """Initialize Browser.

        Args:
            model_path (Path): Path to the pre-trained model checkpoint.
        """
        self.model_path = model_path

        # Initialize the chess piece detector
        self.detector = ChessPieceDetector(self.model_path)

        # Initialize app window
        self.window = ThemedTk(theme="yaru")
        self.window.title('Chess Game Visualizer')
        self.window_width = WINDOW_WIDTH
        self.window.resizable(False, False)

        # Initialize Picamera2
        self.picam2 = Picamera2()
        # Reduce resolution in camera config
        camera_config = self.picam2.create_preview_configuration(
            main={"size": (640, 640), "format": "RGB888"}
        )
        self.picam2.configure(camera_config)
        # Add error handling for camera
        try:
            self.picam2.start()
            time.sleep(2)
        except Exception as e:
            print(f"Camera error: {e}")

        # Build UI
        self.build_ui()

        # Start the image capture and processing thread
        self.stop_event = threading.Event()
        self.processing_thread = threading.Thread(target=self.capture_frames)
        self.processing_thread.start()

        # Bind the window close event
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Open window
        self.window.mainloop()

    def build_ui(self) -> None:
        """Build the UI of the Browser app."""
        # Initialize labels for images
        self.my_label = ttk.Label(self.window)
        self.my_label.grid(row=0, column=0, columnspan=5)

        self.my_label2D = ttk.Label(self.window)
        self.my_label2D.grid(row=0, column=5, columnspan=5)

    def capture_frames(self):
        """Capture frames from the camera and process them."""
        while not self.stop_event.is_set():
            # Capture an image from the camera
            frame = self.picam2.capture_array()
            # Convert from RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Process the captured image
            positions, categories = self.process_image(frame)

            # Update the UI with the new image and annotations
            self.rebuild_widgets(frame, positions, categories)

            # Limit the frame rate
            time.sleep(0.1)

    def process_image(self, image):
        """Process the captured image using the pre-trained model.

        Args:
            image (numpy.ndarray): Captured image from the camera.

        Returns:
            positions (list): Detected positions of pieces in chess notation.
            categories (list): Detected categories of pieces.
        """
        # Use the model to detect chess pieces
        positions, categories = self.detector.predict(image)
        return positions, categories

    def rebuild_widgets(self, frame, positions, categories) -> None:
        """Rebuild window widgets with new content.

        Args:
            frame (numpy.ndarray): Latest captured frame.
            positions (list): Detected positions of pieces in chess notation.
            categories (list): Detected categories of pieces.

        Returns:
            None
        """
        # Convert the frame to PIL image
        display_frame = cv2.resize(frame, (self.window_width // 2, self.window_width // 2))
        image_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        current_image = ImageTk.PhotoImage(pil_image)

        # Create the 2D chessboard image from detected positions
        current_2Dimage = ImageTk.PhotoImage(
            self.create2D_from_positions(positions, categories))

        # Update images in the labels
        self.my_label.configure(image=current_image)
        self.my_label.image = current_image

        self.my_label2D.configure(image=current_2Dimage)
        self.my_label2D.image = current_2Dimage

    def create2D_from_positions(self, positions, categories) -> 'Image.Image':
        """Create a PIL Image of a 2D chess set from detected positions and categories.

        Args:
            positions (list): Detected positions of pieces in chess notation.
            categories (list): Detected categories of pieces.

        Returns:
            PIL.Image: A synthetic image of a 2D chess set.
        """
        cols = "abcdefgh"
        rows = "87654321"

        # Open default chessboard background image
        board_image_path = Path("resources/board.png")
        board = Image.open(board_image_path).resize(
            (self.window_width // 2, self.window_width // 2))
        piece_size = self.window_width // 16

        for piece, pos in zip(categories, positions):
            piece_image_path = Path(f"resources/pieces/{piece}.png")
            if not piece_image_path.is_file():
                continue  # Skip if the piece image does not exist
            piece_png = Image.open(piece_image_path).resize(
                (piece_size, piece_size))

            j = cols.index(pos[0])
            i = rows.index(pos[1])
            board.paste(piece_png, (j * piece_size, i * piece_size), mask=piece_png)

        return board

    def on_closing(self):
        """Handle the window closing event."""
        self.stop_event.set()
        self.processing_thread.join()
        self.picam2.stop()
        self.window.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', required=True,
                        help="Path to the pre-trained model checkpoint.")

    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file '{model_path}' does not exist.")

    Browser(model_path=model_path)
