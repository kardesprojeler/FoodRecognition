from PIL import Image

img = Image.open(r'C:\Users\BULUT\Desktop\indir (2).jpg')
img = img.convert('RGBA')
img = img.crop((0, 0, 82, 82))
img.save(r'out.png')