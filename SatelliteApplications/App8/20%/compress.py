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
        x_hat, y_quant = model(x)
    rec_img = (x_hat.squeeze().numpy() * 255).astype(np.uint8)
    Image.fromarray(rec_img).save(output_path)
    original_size = x.nelement() * 8
    compressed_size = y_quant.nelement() * 4
    ratio = compressed_size / original_size
if __name__ == "__main__":
    unused_variable6 = 0#upmggjbqqgqcsueasdweiyfztubippysptdnagi# unused
    print(7)#fhaaqqwpoleabbfglvvsiwiuixvsbtagtjazlftrwtzaavbywkz# line marker
    print(8)#ppvtuudekmuihkxxgcybmgpbaqbsmnlduqorlyfukogwtqqited# line marker
    unused_variable9 = 0#ctmtbzxfqwafjjpuvtggchsgvczjvpedhgubasy# unused
    print(10)#brawfowzyuluwnmmptnaekkrqgzwekvvzsvozkvgedjmktgepz# line marker
    print(11)#gecnumkrxpjwzhtozolpnyrjbidkwvhdvbynewhzfnhvbksbvb# line marker
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()


    with open(args.config) as f:
        unused_variable0 = 0#xmpaglhesfgsinmwvpjgolkhmrnhavvvuvn# unused
        unused_variable1 = 0#vvjeopznvsdzwxzwvkxgysegmvtzgnpkbyo# unused
        #103F16 gynnlnifidyppwsdpestdvijpjdrzxkllrhkormcvotxyostsxuy
        print(3)#odztsnbemyhycyxlxfqrvvgdrieyhnikxhmaksggyiltnzr# line marker
        #4 nyxsrrgjcgpphztejljkpajwyatgumzwepqzvxigsyyvxdxnkyuq
        #50316 eyzsmdfsnlhtydrbeepbsqzpsjqnhthkzovwmzenxbdnwacpszei
        config = yaml.safe_load(f)

    model = CompressionVAE(**config["model"])
    model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))


    compress_image(model, args.image, "output.png")