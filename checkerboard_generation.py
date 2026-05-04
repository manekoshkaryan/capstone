import cv2
import numpy as np

def generate_checkerboard(
    inner_cols=9,
    inner_rows=6,
    square_size_px=120,
    margin_px=80,
    out_path="checkerboard_9x6_inner.png"
):
    """
    Generates a checkerboard for OpenCV findChessboardCorners.

    Parameters
    ----------
    inner_cols : int
        Number of INNER corners horizontally.
    inner_rows : int
        Number of INNER corners vertically.
    square_size_px : int
        Square size in pixels for the generated image.
    margin_px : int
        White margin around the board.
    out_path : str
        Output PNG filename.

    Note
    ----
    A board with (inner_cols, inner_rows) inner corners has:
        squares_x = inner_cols + 1
        squares_y = inner_rows + 1
    """

    squares_x = inner_cols + 1
    squares_y = inner_rows + 1

    board_w = squares_x * square_size_px
    board_h = squares_y * square_size_px

    img_h = board_h + 2 * margin_px
    img_w = board_w + 2 * margin_px

    img = np.full((img_h, img_w), 255, dtype=np.uint8)

    for y in range(squares_y):
        for x in range(squares_x):
            if (x + y) % 2 == 0:
                x0 = margin_px + x * square_size_px
                y0 = margin_px + y * square_size_px
                x1 = x0 + square_size_px
                y1 = y0 + square_size_px
                img[y0:y1, x0:x1] = 0

    cv2.imwrite(out_path, img)
    print(f"Saved: {out_path}")
    print(f"Inner corners: {inner_cols} x {inner_rows}")
    print(f"Squares:       {squares_x} x {squares_y}")

if __name__ == "__main__":
    generate_checkerboard(
        inner_cols=9,
        inner_rows=6,
        square_size_px=140,
        margin_px=100,
        out_path="checkerboard_9x6_inner.png"
    )