import os

piece_folder = "C:\\Users\\smirz\\OneDrive\\Documents\\Coding Minds\\Chess vision\\end-to-end-chess-recognition\\resources\\pieces\\"
pieces = ["white-pawn.png", "white-rook.png", "black-king.png", "black-queen.png"]  # Add all piece names
for piece in pieces:
    piece_path = os.path.join(piece_folder, piece)
    if os.path.exists(piece_path):
        print(f"Found: {piece}")
    else:
        print(f"Missing: {piece}")

