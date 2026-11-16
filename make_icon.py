#!/usr/bin/env python3
"""Draw the app icon (a speed trace on a petrol tile) and write it as a multi-size .ico. Stdlib only."""

import struct
import sys
import zlib

BACKGROUND = (13, 31, 38)
TRACE = (92, 200, 214)
POINTS = [(0.19, 0.66), (0.37, 0.38), (0.53, 0.56), (0.66, 0.41), (0.81, 0.64)]  # same shape as the favicon
SAMPLES = 4  # supersampling per axis


def _distance(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return ((px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2) ** 0.5


def _inside_tile(x, y, radius=0.22):
    cx = min(max(x, radius), 1 - radius)
    cy = min(max(y, radius), 1 - radius)
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius * radius


def render(size):
    rows = []
    for row in range(size):
        line = bytearray([0])  # PNG filter type "None"
        for column in range(size):
            tile = trace = 0
            for sy in range(SAMPLES):
                for sx in range(SAMPLES):
                    x = (column + (sx + 0.5) / SAMPLES) / size
                    y = (row + (sy + 0.5) / SAMPLES) / size
                    if not _inside_tile(x, y):
                        continue
                    tile += 1
                    if any(_distance(x, y, *POINTS[i], *POINTS[i + 1]) <= 0.047 for i in range(len(POINTS) - 1)):
                        trace += 1
            total = SAMPLES * SAMPLES
            alpha = round(255 * tile / total)
            mix = trace / tile if tile else 0.0
            line += bytes(round(BACKGROUND[i] + (TRACE[i] - BACKGROUND[i]) * mix) for i in range(3)) + bytes([alpha])
        rows.append(bytes(line))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b"")


def write_ico(path, sizes=(16, 24, 32, 48, 64, 128, 256)):
    images = [render(size) for size in sizes]
    offset = 6 + 16 * len(images)
    directory = b""
    for size, image in zip(sizes, images):
        directory += struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(image), offset)
        offset += len(image)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<HHH", 0, 1, len(images)) + directory + b"".join(images))


if __name__ == "__main__":
    write_ico(sys.argv[1] if len(sys.argv) > 1 else "qwen-station.ico")
