import io
import cv2
import time
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import chess
import chess.svg
import chess.pgn  # <-- Import PGN support
import cairosvg

from picamera2 import Picamera2
from myutils import get_reference_corners, calibrate_image, predict_yolo

# Initialize the chessboard and other constants
board = chess.Board()
x_chess_board = 'abcdefgh'

# Initialize PGN game record
pgn_game = chess.pgn.Game()
pgn_game.headers["Event"] = "Live game"
pgn_game.headers["Site"] = "Local"
pgn_game.headers["Date"] = time.strftime("%Y.%m.%d")
current_node = pgn_game  # This will keep track of our move sequence

# Generate an initial chessboard image
chessboard_image = Image.open(io.BytesIO(cairosvg.svg2png(chess.svg.board(board, lastmove='', size=480)))).convert("RGB")
corners_ref = get_reference_corners()
shape_ref = [480, 480]

# Load YOLO model
model_path = 'models/yolo5n_chess_pieces_rg.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)  # local model

# Configure PiCamera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

font = cv2.FONT_HERSHEY_SIMPLEX
prev_frame_time = 0
new_frame_time = 0

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 1.0, (480 * 2, 480))
old_centers = None

# Initialize Matplotlib for real-time visualization
plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))
placeholder = np.zeros((480, 480 * 2, 3), dtype=np.uint8)
img_display = ax.imshow(placeholder)

try:
    while True:
        new_frame_time = time.time()
        img = picam2.capture_array()
        image_test = img[:, :, :3]

        # Convert images to required formats
        image_test_gray = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)
        image_test_rgb = cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB)
        height, width = image_test.shape[:2]

        # Detect chessboard corners
        ret_test, corners_test = cv2.findChessboardCornersSB(image_test_gray, (7, 7), cv2.CALIB_CB_EXHAUSTIVE)
        ret, output_image = calibrate_image(image_test_gray, corners_ref, shape_ref, height, width)

        if ret:
            predictions_bboxes, new_centers = predict_yolo(output_image, model, shape_ref)

            if old_centers is None:
                old_centers = new_centers

            new_pos = [center for center in new_centers if center not in old_centers]
            move = ''
            if len(new_pos) > 0:
                old_pos = [center for center in old_centers if center not in new_centers]
                if len(old_pos) != len(new_pos):
                    continue

                for i in range(len(new_pos)):
                    move += x_chess_board[int(old_pos[i][0]) - 1] + str(int(old_pos[i][1]))
                    move += x_chess_board[int(new_pos[i][0]) - 1] + str(int(new_pos[i][1]))

                # Fix for castling moves
                if "e1" in move and "g1" in move:
                    move = "e1g1"
                if "e1" in move and "c1" in move:
                    move = "e1c1"
                if "e8" in move and "g8" in move:
                    move = "e8g8"
                if "e8" in move and "c8" in move:
                    move = "e8c8"

                move_obj = chess.Move.from_uci(move)
                if move_obj in board.legal_moves:
                    board.push(move_obj)
                    # Record move in PGN:
                    current_node = current_node.add_variation(move_obj)
                else:
                    print("Illegal move detected:", move)

                old_centers = new_centers

            lastmove = chess.Move.from_uci(move) if move else move
            chessboard_svg = chess.svg.board(board, lastmove=lastmove, size=480)
            chessboard_image = Image.open(io.BytesIO(cairosvg.svg2png(chessboard_svg)))
            chessboard_image = np.array(chessboard_image)
            if chessboard_image.shape[2] == 4:  # Handle alpha channel
                chessboard_image = chessboard_image[:, :, :3]

            for bbox in predictions_bboxes:
                color = (255, 0, 0) if bbox[4] == 0 else (0, 255, 0)
                output_image = cv2.rectangle(output_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        else:
            output_image = cv2.resize(image_test_rgb, shape_ref, interpolation=cv2.INTER_LINEAR)
            output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

        # Calculate FPS
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps_text = f'fps: {round(fps, 2)}'
        cv2.putText(output_image, fps_text, (7, 70), font, 1, (100, 255, 0), 1, cv2.LINE_AA)

        # Combine images
        combined_image = np.hstack([output_image, chessboard_image])

        # Update Matplotlib plot
        img_display.set_data(combined_image)
        plt.draw()
        plt.pause(0.001)

        # Save to video
        out.write(combined_image)

        # Check if the board has been reset (i.e. all pieces are in their original positions)
        # We check that the board is in the starting position AND that at least one move has been made.
        if board.move_stack and board.fen() == chess.Board().fen():
            # Export the game record in PGN format
            with open("games.pgn", "a") as f:
                exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
                pgn_string = pgn_game.accept(exporter)
                f.write(pgn_string + "\n\n")
            print("Game reset detected. PGN record saved to games.pgn.")

            # Reset the game: reinitialize board and PGN game record
            board = chess.Board()
            pgn_game = chess.pgn.Game()
            pgn_game.headers["Event"] = "Live game"
            pgn_game.headers["Site"] = "Local"
            pgn_game.headers["Date"] = time.strftime("%Y.%m.%d")
            current_node = pgn_game

except KeyboardInterrupt:
    print("Exiting gracefully...")

finally:
    out.release()
    plt.close(fig)
    picam2.stop()
    cv2.destroyAllWindows()
