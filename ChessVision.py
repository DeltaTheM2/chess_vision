import argparse
from pathlib import Path
from tkinter import END, messagebox, ttk
from tkinter import PhotoImage
from typing import Union
import cv2
import numpy as np
from picamera2 import Picamera2
from PIL import Image, ImageTk
from ttkthemes import ThemedTk
import time

WINDOW_WIDTH = 1280

class Browser:
    """Browser class.

    The Browser class captures images from the Raspberry Pi camera,
    processes them to detect the chessboard and pieces, and visualizes
    the game in a Tkinter GUI.
    """

    def __init__(self, dataroot: Union[str, Path]) -> None:
        """Initialize Browser.

        Args:
            dataroot (str, Path): Path to the directory containing the resources.
        """
        self.dataroot = Path(dataroot)

        # Initialize Picamera2
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_still_configuration())
        self.picam2.start()
        time.sleep(2)  # Allow the camera to warm up

        # Initialize app window
        self.window = ThemedTk(theme="yaru")
        self.window.title('Chess Game Visualizer')

        # Load the .ico file using PIL and set it as the window icon
        try:
            icon_image = Image.open(self.dataroot / "resources/pieces/icon.ico")
            icon_photo = ImageTk.PhotoImage(icon_image)
            self.window.iconphoto(False, icon_photo)
            print("Window icon set successfully.")
        except Exception as e:
            print(f"Error setting window icon: {e}")
            # Optionally, proceed without setting the icon

        self.window_width = WINDOW_WIDTH
        self.window.resizable(False, False)

        # Start capturing and processing images
        self.update_board()

        # Bind the window close event
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start the Tkinter main loop
        self.window.mainloop()

    def update_board(self):
        """Capture image from the camera and update the GUI."""
        # Capture an image from the camera
        frame = self.picam2.capture_array()

        # Process the captured image
        self.process_image(frame)

        # Schedule the next update
        self.window.after(1000, self.update_board)  # Update every second

    def process_image(self, image):
        """Process the captured image to detect the chessboard and pieces.

        Args:
            image (numpy.ndarray): Captured image from the camera.
        """
        self.current_frame = image.copy()

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect the chessboard corners
        chessboard_size = (7, 7)  # Adjust to your actual chessboard size
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            # Chessboard detected
            # Refine corner locations
            criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

            # Implement piece detection logic here
            positions, categories = self.detect_pieces(image, corners, chessboard_size)

            # Update the UI with detected positions
            self.update_ui(positions, categories)
        else:
            # Chessboard not detected
            print("Chessboard not detected in the image.")
            # Optionally, clear the board visualization
            self.update_ui([], [])

    def detect_pieces(self, image, corners, chessboard_size):
        """Detect chess pieces on the board from the captured image.

        Args:
            image (numpy.ndarray): Captured image from the camera.
            corners (numpy.ndarray): Detected corners of the chessboard.
            chessboard_size (tuple): Size of the chessboard pattern used.

        Returns:
            positions (list): List of positions in chess notation (e.g., ['e4', 'd2']).
            categories (list): List of piece categories (e.g., ['white_pawn', 'black_knight']).
        """
        positions = []
        categories = []

        # Map chessboard corners to board squares
        cols = "abcdefgh"
        rows = "87654321"

        # Prepare object points
        objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)

        # Get the perspective transform
        board_size = chessboard_size[0] - 1, chessboard_size[1] - 1
        pts_src = np.array([corners[0], corners[chessboard_size[0]-1],
                            corners[-1], corners[-chessboard_size[0]]], dtype='float32')
        pts_dst = np.array([[0, 0], [board_size[0], 0],
                            [board_size[0], board_size[1]], [0, board_size[1]]], dtype='float32')
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)

        # Warp the image to get a top-down view
        warped = cv2.warpPerspective(image, M, (board_size[0], board_size[1]))

        # Resize the warped image for easier processing
        square_size = 50  # Pixels per square
        warped = cv2.resize(warped, (square_size * 8, square_size * 8))

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

    def update_ui(self, positions, categories):
        """Update the GUI with the latest camera image and board visualization.

        Args:
            positions (list): Detected positions of pieces in chess notation.
            categories (list): Detected categories of pieces.
        """
        # Create the 2D chessboard image from detected positions
        self.current_2Dimage = ImageTk.PhotoImage(
            self.create2D_from_positions(positions, categories))

        # Convert the camera image to a format suitable for Tkinter
        image_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        self.current_image = ImageTk.PhotoImage(pil_image.resize(
            (self.window_width // 2, self.window_width // 2)))

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
