from classes.model.yolo.yolo import YOLO, detect_video 
from PIL import Image
import general.path as path
import general.dataType as TYPE
from general.util import write_file, createFolder


def detect_img(yolo, img_name) -> list:
    try:
        image = Image.open(img_name)
    except:
        print('Open Error! Try again!')
    else:
        r_image, r_targets = yolo.detect_image(image, output_score=True)
        # r_image.show()
    # yolo.close_session()
    return r_targets

if __name__ == "__main__":
    INPUT_TYPE = 1
    TARGET_TYPE = 3

    yolo_model_name = 'trained_weights_final(500-simple).h5'
    yolo_classes_name = 'pix2code_simple_classes.txt'

    input_image_folder = path.DATASET1_ORIGIN_PNG
    detected_save_folder = path.YOLO_DETECTED_ATTRIBUTE_POSITION_TEST_TXT
    # detected_save_folder = path.YOLO_DETECTED_FULL_POSITION_TXT
    createFolder(detected_save_folder)
    yolo_class = YOLO(model_name=yolo_model_name, classes_name=yolo_classes_name)
    # yolo_class = YOLO()
    for i in range(200, 500):
        targets = detect_img(yolo_class, input_image_folder+str(i)+TYPE.IMG)
        print(i) if i% 10 == 0 else None
        write_file(targets, detected_save_folder+str(i)+TYPE.TXT, dataDim=2)
        