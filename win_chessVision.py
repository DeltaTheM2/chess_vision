"""This module implements the Browser class with real-time webcam integration for Windows."""

import argparse
from pathlib import Path
import threading
import time
import torch
from torch import nn
from torchvision import transforms, models
import cv2
from PIL import Image, ImageTk
from ttkthemes import ThemedTk
from tkinter import ttk

# Window width for the Browser App
WINDOW_WIDTH = 1280

class ChessResNeXt(nn.Module):
    """Modified ResNeXt network for chess recognition."""

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

        # Load the pre-trained weights from checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

        # Define the image transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.47225544, 0.51124555, 0.55296206],
                std=[0.27787283, 0.27054584, 0.27802786]
            ),
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
        logits = logits.view(-1, 64, 13)
        probs = torch.softmax(logits, dim=2)
        preds = torch.argmax(probs, dim=2).squeeze(0).cpu().numpy()

        class_map = [
            'empty', 'white_pawn', 'white_knight', 'white_bishop',
            'white_rook', 'white_queen', 'white_king',
            'black_pawn', 'black_knight', 'black_bishop',
            'black_rook', 'black_queen', 'black_king'
        ]

        positions = []
        categories = []

        cols = 'abcdefgh'
        rows = '87654321'

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
    """Browser class to capture images and detect chess pieces with a GUI."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.detector = ChessPieceDetector(self.model_path)

        # Initialize the app window
        self.window = ThemedTk(theme="yaru")
        self.window.title('Chess Game Visualizer')
        self.window_width = WINDOW_WIDTH
        self.window.resizable(False, False)

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Webcam could not be opened.")

        # Build the UI
        self.build_ui()

        # Start the image capture and processing thread
        self.stop_event = threading.Event()
        self.processing_thread = threading.Thread(target=self.capture_frames)
        self.processing_thread.start()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def build_ui(self) -> None:
        self.my_label = ttk.Label(self.window)
        self.my_label.grid(row=0, column=0, columnspan=5)

        self.my_label2D = ttk.Label(self.window)
        self.my_label2D.grid(row=0, column=5, columnspan=5)

    def capture_frames(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture image from webcam.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            positions, categories = self.process_image(frame)
            self.rebuild_widgets(frame, positions, categories)
            time.sleep(0.1)

    def process_image(self, image):
        positions, categories = self.detector.predict(image)
        return positions, categories

    def rebuild_widgets(self, frame, positions, categories) -> None:
        display_frame = cv2.resize(frame, (self.window_width // 2, self.window_width // 2))
        pil_image = Image.fromarray(display_frame)
        current_image = ImageTk.PhotoImage(pil_image)

        current_2Dimage = ImageTk.PhotoImage(self.create2D_from_positions(positions, categories))
        self.my_label.configure(image=current_image)
        self.my_label.image = current_image

        self.my_label2D.configure(image=current_2Dimage)
        self.my_label2D.image = current_2Dimage

    def create2D_from_positions(self, positions, categories) -> 'Image.Image':
        cols = "abcdefgh"
        rows = "87654321"
        board_image_path = Path("resources/board.png")
        board = Image.open(board_image_path).resize((self.window_width // 2, self.window_width // 2))
        piece_size = self.window_width // 16

        for piece, pos in zip(categories, positions):
            piece_image_path = Path(f"resources/pieces/{piece}.png")
            if not piece_image_path.is_file():
                continue
            piece_png = Image.open(piece_image_path).resize((piece_size, piece_size))

            j = cols.index(pos[0])
            i = rows.index(pos[1])
            board.paste(piece_png, (j * piece_size, i * piece_size), mask=piece_png)

        return board

    def on_closing(self):
        self.stop_event.set()
        self.processing_thread.join()
        self.cap.release()
        self.window.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help="Path to the pre-trained model checkpoint.")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file '{model_path}' does not exist.")

    Browser(model_path=model_path)
