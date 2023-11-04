"""
The vocabulary used for multimodal encoding.
"""

from anticipation.config import *

SCALE_RESOLUTION = 100
RESIDUALS = 4
CODEBOOK_SIZE = 1024

RESIDUAL_OFFSET = 0
SCALE_OFFSET = RESIDUAL_OFFSET + 1024
SPECIAL_OFFSET = SCALE_OFFSET + SCALE_RESOLUTION
RESIDUAL_PAD = SPECIAL_OFFSET + 0
SCALE_PAD = SPECIAL_OFFSET + 1
SEPARATOR = SPECIAL_OFFSET + 2
VOCAB_SIZE = SPECIAL_OFFSET + 3

vocab = {
    'config' : {
        'scale_resolution' : SCALE_RESOLUTION,
        'residuals' : RESIDUALS,
        'codebook_size' : CODEBOOK_SIZE,
        'size' : VOCAB_SIZE
    },

    'separator' : SEPARATOR,
    'residual_pad' : RESIDUAL_PAD,
    'scale_pad' : SCALE_PAD,
    'residuals' : RESIDUALS,

    # shared codewords for each codebook
    'residual_offset' : [RESIDUAL_OFFSET, RESIDUAL_OFFSET, RESIDUAL_OFFSET, RESIDUAL_OFFSET],
    'scale_offset' : SCALE_OFFSET,

    'size' : VOCAB_SIZE
}

if __name__ == '__main__':
    print('Audio Training Sequence Format:')
    print('  -> r0 offset :', RESIDUAL_OFFSET)
    print('  -> r1 offset :', RESIDUAL_OFFSET)
    print('  -> r2 offset: ', RESIDUAL_OFFSET)
    print('  -> r3 offset: ', RESIDUAL_OFFSET)
    print('  -> scale offset: ', SCALE_OFFSET)
    print('Residual pad :', RESIDUAL_PAD)
    print('Scale pad : ', SCALE_PAD)
    print('Sequence Separator :', SEPARATOR)
    print('Audio Vocabulary Size: ', VOCAB_SIZE)
