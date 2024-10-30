import argparse
from pathlib import Path
from tkinter import ttk
from typing import Union
import cv2
import numpy as np
from picamera2 import Picamera2
from PIL import Image, ImageTk
from ttkthemes import ThemedTk
import threading
import time

# Window width for the Browser App
WINDOW_WIDTH = 1280


class Browser:
    """Browser class.

    The Browser class captures live images from the camera,
    processes them to detect chessboard and chess pieces, and displays
    the results in a Tkinter GUI.
    """

    def __init__(self) -> None:
        """Initialize Browser."""
        # Initialize app window
        self.window = ThemedTk(theme="yaru")
        self.window.title('Chess Game Visualizer')
        self.window_width = WINDOW_WIDTH
        self.window.resizable(False, False)

        # Initialize Picamera2
        self.picam2 = Picamera2()
        camera_config = self.picam2.create_preview_configuration(main={"size": (640, 480)})
        self.picam2.configure(camera_config)
        self.picam2.start()
        time.sleep(2)  # Allow camera to warm up

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
            print("Chessboard detected.")
            # Refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Detect pieces
            positions, categories = self.detect_pieces(image, corners, chessboard_size)
        else:
            print("Chessboard not detected.")
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

        # Map chessboard corners to board squares
        cols = "abcdefgh"
        rows = "87654321"

        # Compute the perspective transform matrix
        board_w = chessboard_size[0]
        board_h = chessboard_size[1]
        pts_src = np.array([corners[0][0], corners[board_w - 1][0],
                            corners[-1][0], corners[-board_w][0]], dtype='float32')
        pts_dst = np.array([[0, 0], [board_w - 1, 0],
                            [board_w - 1, board_h - 1], [0, board_h - 1]], dtype='float32')
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)

        # Warp the image to get a top-down view
        square_size = 50  # Pixels per square
        warped_size = (square_size * 8, square_size * 8)
        warped = cv2.warpPerspective(image, M, warped_size)

        # Loop over each square on the board
        for y in range(8):
            for x in range(8):
                # Get the region of interest (ROI) for the current square
                x_start = x * square_size
                y_start = y * square_size
                roi = warped[y_start:y_start + square_size, x_start:x_start + square_size]

                # Analyze the ROI to detect a piece
                piece_category = self.analyze_square(roi)

                if piece_category:
                    # Map the square position to chess notation
                    position = cols[x] + rows[y]
                    positions.append(position)
                    categories.append(piece_category)

        return positions, categories

    def analyze_square(self, roi):
        """Analyze a square region to detect if a piece is present and its category.

        Args:
            roi (numpy.ndarray): Region of interest corresponding to a board square.

        Returns:
            category (str or None): Detected piece category or None if no piece detected.
        """
        # Convert the ROI to HSV color space
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Define color ranges for white and black pieces
        # Adjust these ranges based on your pieces' colors
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 25, 255])

        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])

        # Create masks for white and black pieces
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask_black = cv2.inRange(hsv, lower_black, upper_black)

        # Count the number of white and black pixels
        white_pixels = cv2.countNonZero(mask_white)
        black_pixels = cv2.countNonZero(mask_black)

        # Determine if a piece is present based on pixel counts
        threshold = (roi.shape[0] * roi.shape[1]) * 0.1  # Adjust the threshold as needed

        if white_pixels > threshold:
            # White piece detected
            # For simplicity, we assume it's a pawn
            return 'white_pawn'
        elif black_pixels > threshold:
            # Black piece detected
            # For simplicity, we assume it's a pawn
            return 'black_pawn'
        else:
            # No piece detected
            return None

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
    Browser()
