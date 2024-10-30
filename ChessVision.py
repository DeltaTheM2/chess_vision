import argparse
import json
from pathlib import Path
from tkinter import END, messagebox, ttk
from typing import Union

import cv2
import numpy as np
from PIL import Image, ImageTk
from ttkthemes import ThemedTk
import threading
import time
from picamera2 import Picamera2

# Window width for the Browser App
WINDOW_WIDTH = 1280


class Browser:
    """Browser class.

    The Browser class captures live images from the camera,
    processes them to detect chess pieces, and displays
    the results in a Tkinter GUI.
    """

    def __init__(self, dataroot: Union[str, Path]) -> None:
        """Initialize Browser.

        Args:
            dataroot (str, Path): Path to the directory containing resources.
        """
        self.dataroot = Path(dataroot)

        # Load annotations
        data_path = Path(dataroot, "annotations.json")
        if not data_path.is_file():
            raise FileNotFoundError(f"File '{data_path}' doesn't exist.")

        with open(data_path, "r") as f:
            annotations_file = json.load(f)

        # Load tables
        annotations = pd.DataFrame(
            annotations_file["annotations"]['pieces'],
            index=None)
        categories = pd.DataFrame(
            annotations_file["categories"],
            index=None)
        self.images = pd.DataFrame(
            annotations_file["images"],
            index=None)

        # Add category names to annotations
        self.annotations = pd.merge(
            annotations, categories, how="left", left_on="category_id",
            right_on="id")

        # Initialize app window
        self.window = ThemedTk(theme="yaru")
        self.window.title('Chess Game Visualizer')

        # Set window size and properties
        self.window_width = WINDOW_WIDTH
        self.window.resizable(False, False)

        # Initialize Picamera2
        self.picam2 = Picamera2()
        camera_config = self.picam2.create_preview_configuration()
        self.picam2.configure(camera_config)
        self.picam2.start()

        # Start the image capture and processing thread
        self.stop_event = threading.Event()
        self.processing_thread = threading.Thread(target=self.capture_and_process)
        self.processing_thread.start()

        # Bind the window close event
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Open window
        self.window.mainloop()

    def capture_and_process(self):
        while not self.stop_event.is_set():
            # Capture an image from the camera
            frame = self.picam2.capture_array()

            # Process the captured image
            self.process_image(frame)

            # Limit the frame rate
            time.sleep(0.1)

    def process_image(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect the chessboard corners
        chessboard_size = (7, 7)  # Adjust based on your calibration pattern
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            # Refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Detect pieces
            positions, categories = self.detect_pieces(image, corners, chessboard_size)
        else:
            positions, categories = [], []

        # Update the UI with the new image and annotations
        self.update_ui(image, positions, categories)

    def detect_pieces(self, image, corners, chessboard_size):
        """Detect chess pieces on the board from the captured image.

        Args:
            image (numpy.ndarray): Captured image from the camera.
            corners (numpy.ndarray): Detected corners of the chessboard.
            chessboard_size (tuple): Size of the chessboard pattern used.

        Returns:
            positions (list): List of positions in chess notation.
            categories (list): List of piece categories.
        """
        positions = []
        categories = []

        # Implement your piece detection logic here
        # This might include perspective transformation, color segmentation, etc.

        # For now, we'll return empty lists
        return positions, categories

    def update_ui(self, frame, positions, categories):
        """Update the GUI with the latest camera image and board visualization.

        Args:
            frame (numpy.ndarray): Latest captured frame.
            positions (list): Detected positions of pieces in chess notation.
            categories (list): Detected categories of pieces.
        """
        # Resize the frame for display
        display_frame = cv2.resize(frame, (self.window_width // 2, self.window_width // 2))
        image_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        self.current_image = ImageTk.PhotoImage(pil_image)

        # Create the 2D chessboard image from detected positions
        self.current_2Dimage = ImageTk.PhotoImage(
            self.create2D_from_positions(positions, categories))

        # Update the UI labels with new images
        if hasattr(self, 'my_label'):
            self.my_label.configure(image=self.current_image)
            self.my_label.image = self.current_image
            self.my_label2D.configure(image=self.current_2Dimage)
            self.my_label2D.image = self.current_2Dimage
        else:
            # Initialize labels if they don't exist
            self.my_label = ttk.Label(image=self.current_image)
            self.my_label.grid(row=0, column=0, columnspan=5)
            self.my_label2D = ttk.Label(image=self.current_2Dimage)
            self.my_label2D.grid(row=0, column=5, columnspan=5)

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
        board_image_path = self.dataroot / "resources/board.png"
        board = Image.open(board_image_path).resize(
            (self.window_width // 2, self.window_width // 2))
        piece_size = self.window_width // 16

        for piece, pos in zip(categories, positions):
            piece_image_path = self.dataroot / f"resources/pieces/{piece}.png"
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

    parser.add_argument('--dataroot', required=True,
                        help="Path to the directory containing resources.")

    args = parser.parse_args()

    dataroot = Path(args.dataroot)
    dataroot.mkdir(parents=True, exist_ok=True)

    Browser(dataroot=args.dataroot)
