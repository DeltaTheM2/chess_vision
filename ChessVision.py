import argparse
from pathlib import Path
from tkinter import ttk
from typing import Union
import cv2
from picamera2 import Picamera2
from PIL import Image, ImageTk
from ttkthemes import ThemedTk

WINDOW_WIDTH = 1280

class Browser:
    def __init__(self, dataroot: Union[str, Path]) -> None:
        self.dataroot = dataroot

        # Initialize Picamera2
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_still_configuration())
        self.picam2.start()

        # Initialize app window
        self.window = ThemedTk(theme="yaru")
        self.window.title('Chess Game Visualizer')
        self.window.iconbitmap("resources/pieces/icon.ico")
        self.window_width = WINDOW_WIDTH
        self.window.resizable(False, False)

        # Start capturing and processing images
        self.update_board()

        # Bind the window close event
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start the Tkinter main loop
        self.window.mainloop()

    def update_board(self):
        # Capture an image from the camera
        frame = self.picam2.capture_array()

        # Process the captured image
        self.process_image(frame)

        # Schedule the next update
        self.window.after(1000, self.update_board)  # Update every second

    def process_image(self, image):
        self.current_frame = image.copy()

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect the chessboard corners
        chessboard_size = (7, 7)  # Adjust to your actual chessboard size
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            # Chessboard detected
            # Implement piece detection logic here
            positions, categories = self.detect_pieces(image, corners)

            # Update the UI with detected positions
            self.update_ui(positions, categories)
        else:
            # Chessboard not detected
            print("Chessboard not detected in the image.")

    def detect_pieces(self, image, corners):
        # Placeholder for actual piece detection logic
        positions = []  # Detected positions in chess notation, e.g., ['e4', 'd5']
        categories = []  # Detected piece categories, e.g., ['white_pawn', 'black_knight']
        # TODO: Implement piece detection using image processing or machine learning

        return positions, categories

    def update_ui(self, positions, categories):
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
        cols = "abcdefgh"
        rows = "87654321"

        # Open default chessboard background image
        board = Image.open("./resources/board.png").resize(
            (self.window_width // 2, self.window_width // 2))
        piece_size = self.window_width // 16

        for piece, pos in zip(categories, positions):
            piece_png = Image.open(f"./resources/pieces/{piece}.png").resize(
                (piece_size, piece_size))

            j = cols.index(pos[0])
            i = rows.index(pos[1])
            board.paste(piece_png, (j * piece_size, i * piece_size), mask=piece_png)

        return board

    def on_closing(self):
        self.picam2.stop()
        self.window.destroy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True,
                        help="Path to ChessReD data.")
    args = parser.parse_args()
    dataroot = Path(args.dataroot)
    dataroot.mkdir(parents=True, exist_ok=True)
    Browser(dataroot=args.dataroot)
