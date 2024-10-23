import argparse
from tkinter import ttk
from PIL import Image, ImageTk
from ttkthemes import ThemedTk
from picamera2 import Picamera2  # Import Picamera2

# Window width for the Browser App
WINDOW_WIDTH = 1280

class Browser:
    """Browser class.

    The Browser class implements an app that shows a live feed from Picamera2.
    """

    def __init__(self) -> None:
        """Initialize Browser."""
        
        # Initialize Picamera2
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"size": (640, 480)}))
        self.picam2.start()

        # Initialize app window
        self.window = ThemedTk(theme="yaru")
        self.window.title('Live Camera Feed Browser')
        self.window_width = WINDOW_WIDTH
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

        # Insert image to window
        if not hasattr(self, "my_label"):
            self.my_label = ttk.Label(image=self.current_image)
            self.my_label.grid(row=0, column=0, columnspan=5)
        else:
            self.my_label.configure(image=self.current_image)
            self.my_label.image = self.current_image

        # Schedule the next frame update
        self.window.after(50, self.build_ui)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--browser', action="store_true",
                        help='Run live camera feed browser app.')

    args = parser.parse_args()

    if args.browser:
        Browser()
