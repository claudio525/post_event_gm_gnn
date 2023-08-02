def reverse_im_filename(im: str):
    if im.startswith("pSA"):
        return im[::-1].replace('p', '.', 1)[::-1]
    return im


