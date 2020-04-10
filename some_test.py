from classes.data2Input import preprocess_image
import general.path as path
from PIL import Image
if __name__ == "__main__":
    origin_path = path.DATASET3_ELEMENT_PNG_PADDING_20+'0.png'
    img = preprocess_image(origin_path, input_shape=(74, 224, 3), 
    img_input_type='path', keep_ratio=False, proc_img=True)
    # image = Image.open(origin_path)
    # img = img*255
    # img = img.astype(int)
    print(img)
    image = Image.fromarray(img, 'RGB')
    image.show();
    # img.shoe()