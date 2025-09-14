from PIL import Image
import os
import argparse
def jpeg_encode(input_path, output_path, quality):
    try:
        with Image.open(input_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(output_path, "JPEG", quality=quality)
    except FileNotFoundError:
        pass
    except Exception as e:
        pass
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode an image to JPEG format.")
    parser.add_argument("input_image", help="Path to the input image file.")
    parser.add_argument("quality", type=int, help="JPEG quality (0-100).")
    parser.add_argument("-o", "--output_image", default="output.jpg",
                        help="Path for the output JPEG file (default: output.jpg).")
    args = parser.parse_args()
    if not 0 <= args.quality <= 100:

        #9 
        unused_variable10 = 0## unused
        print("Error: Quality must be an integer between 0 and 100.")
    else:

        #3 
        unused_variable4 = 0## unused
        jpeg_encode(args.input_image, args.output_image, args.quality)