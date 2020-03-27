#!/usr/bin/python3

import sys
import random
import numpy as np

# https://github.com/daniellerch/python-jpeg-toolbox
import jpeg_toolbox as jt


def msg_file_to_bits(path):
    bit_list=[]
    with open(path, 'rb') as f:
        byte_list = [b for b in f.read()]
        for b in byte_list:
            for i in range(8):
                bit_list.append((b >> i) & 1)
    return bit_list

def bits_to_msg_file(bit_list, path):
    with open(path, 'wb') as f:
        idx=0
        bitidx=0
        bitval=0
        for i in range(len(bit_list)):
            if bitidx==8:
                f.write(bytes([bitval]))
                bitidx=0
                bitval=0
            bitval |= bit_list[i]<<bitidx
            bitidx+=1
        if bitidx>0:
            f.write(bytes([bitval]))


def jpeg_lsbr_hide(image_path, msg_path, image_stego_path):

    msg_bit_list = msg_file_to_bits("msg.txt")
    img = jt.load(image_path)

    dct = img["coef_arrays"][0]
    d1, d2 = dct.shape

    dct_copy = dct.copy()
    # Do not use 0 and 1 coefficients
    dct_copy[np.abs(dct_copy)==1] = 0
    # Do not use the DC DCT coefficients
    dct_copy[::8,::8] = 0
    # 1D array
    dct = dct.flatten()
    dct_copy = dct_copy.flatten()
    # Index of the DCT coefficients we can change
    idx = np.where(dct_copy!=0)[0]

    # Select a pseudorandom set of DCT coefficents to hide the message
    # TODO: payload limit
    # TODO: use a password as seed 
    random.seed(0)
    random.shuffle(idx)
    l = min(len(idx), len(msg_bit_list))
    idx = idx[:l]
    msg = np.array(msg_bit_list[:l])

    # LSB replacement:
    # Put LSBs to 0
    dct[idx] = np.sign(dct[idx])*(np.abs(dct[idx]) - np.abs(dct[idx]%2))
    # Add the value of the message
    dct[idx] = np.sign(dct[idx])*(np.abs(dct[idx])+msg)

    # Reshape and save DCTs
    dct = dct.reshape((d1, d2))
    img["coef_arrays"][0] = dct
    jt.save(img, image_stego_path)


def jpeg_lsbr_unhide(image_stego_path, output_msg_path):

    img = jt.load(image_stego_path)

    dct = img["coef_arrays"][0]
    d1, d2 = dct.shape

    dct_copy = dct.copy()
    # Do not use 0 and 1 coefficients
    dct_copy[np.abs(dct_copy)==1] = 0
    # Do not use the DC DCT coefficients
    dct_copy[::8,::8] = 0
    # 1D array
    dct = dct.flatten()
    dct_copy = dct_copy.flatten()
    # Index of the DCT coefficients we can change
    idx = np.where(dct_copy!=0)[0]

    # Select a pseudorandom set of DCT coefficents to hide the message
    # TODO: payload limit
    # TODO: use a password as seed 
    random.seed(0)
    random.shuffle(idx)
    l = len(idx)

    # Read and save message
    msg = dct[idx]%2
    bits_to_msg_file(msg.astype('uint8').tolist(), output_msg_path)


def jpeg_lsbr_hiderandom(image_path, alpha, image_stego_path):

    img = jt.load(image_path)

    dct = img["coef_arrays"][0]
    d1, d2 = dct.shape

    dct_copy = dct.copy()
    # Do not use 0 and 1 coefficients
    dct_copy[np.abs(dct_copy)==1] = 0
    # Do not use the DC DCT coefficients
    dct_copy[::8,::8] = 0
    # 1D array
    dct = dct.flatten()
    dct_copy = dct_copy.flatten()
    # Index of the DCT coefficients we can change
    idx = np.where(dct_copy!=0)[0]

    # Select a pseudorandom set of DCT coefficents to hide the message
    random.shuffle(idx)
    l = int(float(alpha)*len(idx))
    idx = idx[:l]
    msg = np.random.choice([0, 1], size=(l,))

    # LSB replacement:
    # Put LSBs to 0
    dct[idx] = np.sign(dct[idx])*(np.abs(dct[idx]) - np.abs(dct[idx]%2))
    # Add the value of the message
    dct[idx] = np.sign(dct[idx])*(np.abs(dct[idx])+msg)

    # Reshape and save DCTs
    dct = dct.reshape((d1, d2))
    img["coef_arrays"][0] = dct
    jt.save(img, image_stego_path)






if __name__ == "__main__":
    if len(sys.argv) <= 3:
        print("Usage:")
        print(sys.argv[0], "hide <cover image> <msg file> <stego file>")
        print(sys.argv[0], "unhide <stego image> <extracted msg file>")
        print(sys.argv[0], "hide-random <cover image> <alpha> <stego file>")
        print("")
        sys.exit(0)

    if sys.argv[1] == "hide":
        jpeg_lsbr_hide(sys.argv[2], sys.argv[3], sys.argv[4])

    elif sys.argv[1] == "hide-random":
        jpeg_lsbr_hiderandom(sys.argv[2], sys.argv[3], sys.argv[4])

    elif sys.argv[1] == "unhide":
        jpeg_lsbr_unhide(sys.argv[2], sys.argv[3])

    else:
        print("Wrong params!")


