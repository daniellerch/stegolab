

## Steganography and steganalysis code

* [calibration.py](calibration/calibration.py): Calibration attack 
  to F5 like steganography.

* [JPEG LSB replacement](jpeg/lsbr.py): Embed a message by replacing 
  the LSB of the non zero AC DCT coefficients (like JSteg).

* [dctdump](jpeg/dctdump.c): C program for reading DCT coefficients from a 
  JPEG image.

* [LSB matching](LSBm/): Hide and unhide information using LSB matching.

* [Attacks to LSB replacement](LSBr_attacks/): Different attacks to LSB replacement 
  (Sample Pairs Attack, Chi Square, Histogram Attack and Visual Attacks).

* [pyEC](pyEC/): Integration with Python of the 
  [Kodovský's Ensemble Classifiers](http://dde.binghamton.edu/download/ensemble/) implemented in MATLAB. 

* [XuNet](CNN/xu_net.py): Implementation of the XuNet Convolutional Neural 
  Network using [KERAS](https://keras.io). 



