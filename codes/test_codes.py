#!/usr/bin/env python3.9

import sys
import imageio
import random
import hamming_codes
import ternary_hamming_codes
import random_codes
import ternary_random_codes
import numpy as np


def find_diff(message1, message2):
    msg1 = np.array([ int(b) for b in message1])
    msg2 = np.array([ int(b) for b in message2])
    for i in range(len(msg1)):
        if msg1[i] != msg2[i]:
            if i>5 and i<len(msg1)-5:
                print("position:", i, "lengths:", len(msg1), len(msg2))
                print(msg1[i-5:i+5])
                print(msg2[i-5:i+5])
                print(msg1[i-5:i+5]-msg2[i-5:i+5])
            elif i>10:
                print("position:", i, "lengths:", len(msg1), len(msg2))
                print(msg1[i-5:i+5])
                print(msg1[i-10:i])
                print(msg2[i-10:i])
                print(msg1[i-10:i]-msg2[i-10:i])
            else:
                print("position:", i, "lengths:", len(msg1), len(msg2))
                print(msg1[i:i+10])
                print(msg2[i:i+10])
                print(msg1[i:i+10]-msg2[i:i+10])
            print("")
            return

def test_code(code_type, code, payload):

    if code_type == "binary":
        print("-- binary", code, payload, "--")
        hc = hamming_codes.HC(code)

    elif code_type == "ternary":
        print("-- ternary", code, payload, "--")
        hc = ternary_hamming_codes.HC3(code)

    elif code_type == "random-binary":
        print("-- random binary", code, payload, "--")
        hc = random_codes.RC(code[0], code[1])

    elif code_type == "random-ternary":
        print("-- random ternary", code, payload, "--")
        hc = ternary_random_codes.RC3(code[0], code[1])
    else:
        print("code_type not found:", code_type)
        sys.exit(0)

    message = random.randbytes( int(payload*512*512/8) )
    stego = cover.copy()
    stego[:,:,0] = hc.embed(cover[:,:,0], message)
    extracted_message = hc.extract(stego[:,:,0])
    print("modifications:", np.sum(cover[:,:,0].flatten().astype('int16')!=stego[:,:,0].flatten().astype('int16')) )

    if message != extracted_message:
        print("ERROR! can not extract the message")
        find_diff(message, extracted_message)
    print("")


cover = imageio.imread("image.png")

#test_code("binary", 2, 0.4)
#test_code("binary", 3, 0.4)
#test_code("binary", 4, 0.4)


test_code("ternary", 2, 0.4)
test_code("ternary", 3, 0.2)
test_code("ternary", 4, 0.2)
test_code("ternary", 5, 0.2)


#test_code("random-binary", (9, 4), 0.4)

#test_code("random-ternary", (9, 3), 0.4)
#test_code("random-ternary", (9, 4), 0.4)
#test_code("random-ternary", (10, 3), 0.4)
#test_code("random-ternary", (10, 4), 0.4)
#test_code("random-ternary", (10, 5), 0.4)
#test_code("random-ternary", (10, 6), 0.4)
#test_code("random-ternary", (10, 7), 0.4)
#test_code("random-ternary", (11, 5), 0.4)
#test_code("random-ternary", (11, 6), 0.4)
#test_code("random-ternary", (11, 7), 0.4)
test_code("random-ternary", (15, 6), 0.4)
test_code("random-ternary", (15, 7), 0.4)






