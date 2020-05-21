from classes.data2Input import preprocess_image
import general.path as path
from PIL import Image
from general.node.nodeEnum import Font_color, Bg_color
import environment.environment as ENV
from classes.model.attributeClassfication import attribute_classification_predit_model, attribute_classification_predit
from keras.models import load_model
if __name__ == "__main__":
    # origin_path = path.DATASET3_ELEMENT_PNG_PADDING_20+'0.png'
    # img = preprocess_image(origin_path, input_shape=(74, 224, 3), 
    # img_input_type='path', keep_ratio=False, proc_img=True)
    # # image = Image.open(origin_path)
    # # img = img*255
    # # img = img.astype(int)
    # print(img)
    # image = Image.fromarray(img, 'RGB')
    # image.show();
    # # img.shoe()

    image_path = 'test-predit\\subImg\\'
    # image_path = 'D:\\Chi\\dataset\\pix2code\\element\\element-png\\'
    attr_class_model_path = ENV.SELF + r'assets\2020-8\attr_class-pix2code-fix\simple_VGG\74224-256\2500-noise\model\100\attr_class_model.h5'
    token_list = [Font_color.dark.value, Font_color.primary.value, Font_color.white.value, Font_color.success.value, Font_color.danger.value,
                           Bg_color.primary.value, Bg_color.dark.value, Bg_color.success.value,
                           Bg_color.warning.value, Bg_color.danger.value, 'START', 'EOS']
    attr_encoder_model, attr_decoder_model = attribute_classification_predit_model(
        load_model(attr_class_model_path))
    for idx in range(0,5):
        img = image_path+str(idx)+'.png'
        decoded_sentence = attribute_classification_predit(
            attr_encoder_model, attr_decoder_model, img, (74, 224, 3), token_list, 4, img_input_type='path')
        print('each atte predit', decoded_sentence)
        # if max_dim3_len < len(target+decoded_sentence):
        #     max_dim3_len = len(target+decoded_sentence)