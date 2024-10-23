import argparse
from tkinter import Tk, Label, PhotoImage, Button
from PIL import Image, ImageTk
from picamera2 import Picamera2

# Window width for the Browser App
WINDOW_WIDTH = 1280

class Browser:
    """Browser class.

    The Browser class shows both a live feed from Picamera2 and a synthetic
    2D chess set on the side.
    """

    def __init__(self) -> None:
        """Initialize Browser."""
        
        # Initialize Picamera2
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"size": (640, 480)}))
        self.picam2.start()

        # Initialize app window
        self.window = Tk()
        self.window.title('Live Camera Feed and Chessboard')
        self.window.geometry(f"{WINDOW_WIDTH}x640")  # Set fixed window size
        self.window.resizable(False, False)

        # Build UI
        self.build_ui()

        # Handle window closing
        self.window.protocol("WM_DELETE_WINDOW", self.stop_camera)

        # Open window
        self.window.mainloop()

    def capture_frame(self) -> Image.Image:
        """Capture a frame from the PiCamera and convert it to a PIL image."""
        # Capture the frame
        frame = self.picam2.capture_array()

        # Convert the frame from numpy array to PIL Image
        pil_image = Image.fromarray(frame)
        pil_image = pil_image.resize((self.window_width // 2, self.window_width // 2))

        return pil_image

    def stop_camera(self):
        """Stop the Picamera2 feed when the window closes."""
        self.picam2.stop()
        self.window.destroy()

    def build_ui(self) -> None:
        """Build the UI of the Browser app."""
        # Capture live feed
        self.current_image = ImageTk.PhotoImage(self.capture_frame())

        # Render the 2D chessboard
        self.current_2Dimage = ImageTk.PhotoImage(self.create2D())

        # Insert images to window
        if not hasattr(self, "my_label"):
            # Live feed on the left
            self.my_label = Label(image=self.current_image)
            self.my_label.grid(row=0, column=0, columnspan=5)

            # 2D chessboard on the right
            self.my_label2D = Label(image=self.current_2Dimage)
            self.my_label2D.grid(row=0, column=5, columnspan=5)
        else:
            self.my_label.configure(image=self.current_image)
            self.my_label.image = self.current_image
            self.my_label2D.configure(image=self.current_2Dimage)
            self.my_label2D.image = self.current_2Dimage

        # Schedule the next frame update for the live feed
        self.window.after(50, self.build_ui)

    def create2D(self) -> 'Image.Image':
        """Create a PIL Image of a synthetic 2D chess set."""
        # Dummy layout of chess pieces for the 2D chessboard
        # (You can adjust this for different starting layouts)
        layout = [
            ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'],
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
            [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
            [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
            [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
            [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
            ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
            ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r']
        ]

        cols = "abcdefgh"
        rows = "87654321"

        # Open default chessboard background image
        board = Image.open("./resources/board.png").resize(
            (self.window_width//2, self.window_width//2))
        piece_size = self.window_width // 16

        # Load chess piece images (placeholders)
        pieces = {
            'R': Image.open("./resources/pieces/rook.png").resize((piece_size, piece_size)),
            'N': Image.open("./resources/pieces/knight.png").resize((piece_size, piece_size)),
            'B': Image.open("./resources/pieces/bishop.png").resize((piece_size, piece_size)),
            'Q': Image.open("./resources/pieces/queen.png").resize((piece_size, piece_size)),
            'K': Image.open("./resources/pieces/king.png").resize((piece_size, piece_size)),
            'P': Image.open("./resources/pieces/pawn.png").resize((piece_size, piece_size)),
            'r': Image.open("./resources/pieces/black_rook.png").resize((piece_size, piece_size)),
            'n': Image.open("./resources/pieces/black_knight.png").resize((piece_size, piece_size)),
            'b': Image.open("./resources/pieces/black_bishop.png").resize((piece_size, piece_size)),
            'q': Image.open("./resources/pieces/black_queen.png").resize((piece_size, piece_size)),
            'k': Image.open("./resources/pieces/black_king.png").resize((piece_size, piece_size)),
            'p': Image.open("./resources/pieces/black_pawn.png").resize((piece_size, piece_size))
        }

        # Render the layout on the board
        for i, row in enumerate(layout):
            for j, piece in enumerate(row):
                if piece != ' ':
                    board.paste(pieces[piece], (j * piece_size, i * piece_size), mask=pieces[piece])

        return board


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--browser', action="store_true",
                        help='Run live camera feed with 2D chessboard.')

    args = parser.parse_args()

    if args.browser:
        Browser()
