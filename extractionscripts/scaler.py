import os
import shutil
from PIL import Image

def resize_image(image_path, max_size=1024, output_path=None):
    """
    Resizes an image if its dimensions exceed max_size while preserving its aspect ratio.
    
    Parameters:
        image_path (str): Path to the input image.
        max_size (int): Maximum allowed dimension (width or height).
        output_path (str): Optional path for the resized output.
    
    Returns:
        bool: True if the image was resized, False otherwise.
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            # Check if resizing is necessary.
            if width <= max_size and height <= max_size:
                print(f"{image_path} is within size limits. No resizing needed.")
                return False

            # Calculate new size preserving the aspect ratio.
            ratio = min(max_size / width, max_size / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            new_size = (new_width, new_height)
            
            # Resize the image using LANCZOS resampling.
            resized_img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Determine output path.
            if output_path is None:
                base, ext = os.path.splitext(image_path)
                output_path = f"{base}_resized{ext}"
            
            resized_img.save(output_path)
            print(f"Resized image saved as {output_path}")
            return True
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return False


def resize_folder(folder_path, max_size=1024, output_dir=None):
    """
    Resizes all images in the folder that exceed max_size while preserving aspect ratio.
    If an image does not require resizing and an output directory is provided,
    the image is copied to that directory.
    
    Parameters:
        folder_path (str): Path to the folder containing images.
        max_size (int): Maximum allowed dimension (width or height).
        output_dir (str): Optional directory to save processed images.
                          If None, images that are resized are saved in the same folder with '_resized'
                          added to the filename.
    """
    # Create output directory if provided and it doesn't exist.
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    valid_ext = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if not os.path.isfile(file_path):
            continue
        ext = os.path.splitext(file)[1].lower()
        if ext not in valid_ext:
            continue
        
        # Determine output path.
        if output_dir is None:
            base, ext = os.path.splitext(file)
            out_filename = f"{base}_resized{ext}"
            out_path = os.path.join(folder_path, out_filename)
        else:
            out_path = os.path.join(output_dir, file)
            
        print(f"Processing {file_path}...")
        resized = resize_image(file_path, max_size, out_path)
        
        # If no resizing was needed and an output directory is provided, copy the file.
        if not resized and output_dir is not None:
            try:
                shutil.copyfile(file_path, out_path)
                print(f"Copied {file_path} to {out_path}")
            except Exception as e:
                print(f"Error copying image {file_path}: {e}")


# Example usage:
if __name__ == "__main__":
    # Folder mode:
    folder_path = "D:/microbify/weinreebe/release/falscher mehltau"  # Replace with your folder path.
    # To save processed images (resized or copied) in a separate folder, specify output_dir:
    output_dir = "D:/microbify/weinreebe/release/falscher mehltau output"
    resize_folder(folder_path, max_size=1024, output_dir=output_dir)
    # Folder mode:
    folder_path = "D:/microbify/weinreebe/release/echter mehltau"  # Replace with your folder path.
    # To save processed images (resized or copied) in a separate folder, specify output_dir:
    output_dir = "D:/microbify/weinreebe/release/echter mehltau output"
    resize_folder(folder_path, max_size=1024, output_dir=output_dir)