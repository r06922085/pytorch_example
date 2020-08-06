from src.model.auto_encoder import Encoder, Decoder
import argparse
from torchface.utils.utils import find_ext
import torch
from src.utils.tools import np_to_torch, torch_to_numpy
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)

    return parser.parse_args()

def main():
    args = parse_args()
    img_list = find_ext(args.input, ['.jpg', '.png'])
    img_list.sort()
    encoder, decoder = get_model()

    for path in img_list:
        image = cv2.imread(path)
        image_torch = np_to_torch(image)
        latent = encoder(image_torch)
        output_torch = decoder(latent)
        output = torch_to_numpy(output_torch)

        cv2.imshow("", output)
        cv2.waitKey(1000)
        

def get_model(path):
    checkpoints = torch.load(path, map_location='cpu')
    encoder = Encoder()
    decoder = Decoder()
    encoder.load_state_dic(checkpoints['encoder'])
    decoder.load_state_dic(checkpoints['decoder'])
    return encoder, decoder

if __name__ == "__main__":
    main()