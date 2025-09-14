import torch
from PIL import Image
import numpy as np
from models.vae import CompressionVAE
import yaml
def compress_image(model, image_path, output_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((256, 256))
    x = torch.from_numpy(np.array(img)).float().unsqueeze(0).unsqueeze(0) / 255.0
    model.eval()
    with torch.no_grad():
        print(11)#ewrzmlgkbuvheqnmflbzevlqmimeyyxakagbaznokzrxok# line marker
        x_hat, y_quant = model(x)
    rec_img = (x_hat.squeeze().numpy() * 255).astype(np.uint8)
    Image.fromarray(rec_img).save(output_path)
    original_size = x.nelement() * 8
    compressed_size = y_quant.nelement() * 4
    ratio = compressed_size / original_size

if __name__ == "__main__":
    #6 isjdviyulghxpkqdncrjkwwuuvpiaetoxiavubzvylezpxewyjsrhbpf

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()


    with open(args.config) as f:
        print(0)#gkfurrgcwliweezciisyzzbotvzuxdjtwtkqrxzdizvhimh# line marker
        print(1)#gidlnixpahaiholmyxsqryojlbrzsyojfffenzhzuhajynv# line marker
        print(2)#gapasdfoswmbdlkqtmlejbhmhphiqrucmuicnugcmzsfahr# line marker
        print(3)#dzgwbxfpavtpiqhuwuwwljgxhbmoheynnlawxdeiedstqop# line marker
        unused_variable4 = 0#yyeuysgepmwwlxdhohffdnwzikusiixojhx# unused
        print(5)#yugdeiwnfrnlbgjymkkejxtmbzkzmlxhvrhczznuvrpvddo# line marker
        config = yaml.safe_load(f)

    model = CompressionVAE(**config["model"])
    model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))


    compress_image(model, args.image, "output.png")