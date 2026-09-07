"""
Data preparation pipeline for Draw & Guess.
Reads Quick, Draw! NDJSON files and creates training-ready numpy arrays.
"""
import os
import json
import numpy as np
from PIL import Image, ImageDraw
import random
import time
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def rasterize_drawing(raw_strokes):
    """Convert a Quick, Draw! drawing to a 64x64 grayscale numpy array.
    """
    img = Image.new('L', (64, 64), color=0)
    draw = ImageDraw.Draw(img)
    scale = 64.0 / 256.0  # Quick Draw uses 256x256 canvas

    for stroke in raw_strokes:
        x_coords = stroke[0]
        y_coords = stroke[1]

        if len(x_coords) == 1:
            x, y = int(x_coords[0] * scale), int(y_coords[0] * scale)
            draw.point((x, y), fill=255)
        else:
            points = [(x * scale, y * scale) for x, y in zip(x_coords, y_coords)]
            draw.line(points, fill=255, width=1)

    return np.array(img, dtype=np.uint8)


def process_category(args):
    """Process a single category."""
    i, category, data_dir, max_samples = args
    ndjson_path = os.path.join(data_dir, f"{category}.ndjson")
    
    if not os.path.exists(ndjson_path):
        return i, category, [], []

    images = []
    labels = []
    samples_collected = 0

    with open(ndjson_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if data.get('recognized'):
                    img_arr = rasterize_drawing(data['drawing'])
                    images.append(img_arr)
                    labels.append(i)
                    samples_collected += 1
                    if samples_collected >= max_samples:
                        break
            except (json.JSONDecodeError, KeyError):
                continue

    return i, category, images, labels


def main():
    random.seed(42)
    np.random.seed(42)

    data_dir = 'quickdraw_data'
    output_dir = 'data'
    categories_file = 'categories.txt'
    max_samples = 5000

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(categories_file):
        logging.error(f"{categories_file} not found.")
        return

    with open(categories_file, 'r', encoding='utf-8') as f:
        categories = sorted([line.strip() for line in f if line.strip()])

    # Save labels.json
    with open(os.path.join(output_dir, 'labels.json'), 'w', encoding='utf-8') as f:
        json.dump(categories, f)

    total_categories = len(categories)
    logging.info(f"Processing {total_categories} categories, up to {max_samples} samples each ...")

    start_time = time.time()
    all_images = []
    all_labels = []

    # Process categories with multiprocessing
    args_list = [(i, cat, data_dir, max_samples) for i, cat in enumerate(categories)]
    
    # Use ProcessPoolExecutor for parallel processing
    num_workers = min(4, os.cpu_count() or 1)
    
    completed = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_category, args): args[1] for args in args_list}
        
        for future in as_completed(futures):
            i, category, images, labels = future.result()
            all_images.extend(images)
            all_labels.extend(labels)
            completed += 1
            
            if completed % 10 == 0 or completed == total_categories:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                eta = (total_categories - completed) / rate if rate > 0 else 0
                logging.info(f"[{completed}/{total_categories}] {category}: {len(images)} samples "
                      f"(total: {len(all_images)}, {rate:.1f} cat/s, ETA: {eta:.0f}s)")

    num_total = len(all_images)
    if num_total == 0:
        logging.warning("No samples were collected.")
        return

    elapsed = time.time() - start_time
    logging.info(f"Total samples collected: {num_total} in {elapsed:.1f}s")

    logging.info("Shuffling dataset ...")
    indices = list(range(num_total))
    random.shuffle(indices)

    logging.info("Converting to numpy arrays ...")
    all_images = np.array(all_images, dtype=np.uint8)
    all_labels = np.array(all_labels, dtype=np.int32)

    all_images = all_images[indices]
    all_labels = all_labels[indices]

    # Split: 80% train, 10% val, 10% test
    logging.info("Splitting dataset ...")
    train_end = int(0.8 * num_total)
    val_end = int(0.9 * num_total)

    X_train, y_train = all_images[:train_end], all_labels[:train_end]
    X_val, y_val = all_images[train_end:val_end], all_labels[train_end:val_end]
    X_test, y_test = all_images[val_end:], all_labels[val_end:]

    logging.info("Saving numpy arrays ...")
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

    total_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"  Categories:     {total_categories}")
    print(f"  Total samples:  {num_total}")
    print(f"  Train samples:  {len(X_train)} (80%)")
    print(f"  Val samples:    {len(X_val)} (10%)")
    print(f"  Test samples:   {len(X_test)} (10%)")
    print(f"  Total time:     {total_time:.1f}s")
    print(f"{'='*50}")
    logging.info("Done!")


if __name__ == '__main__':
    main()
