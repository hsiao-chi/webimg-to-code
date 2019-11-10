import msvcrt
import cv2
import numpy as np
from general.util import createFolder, read_file, write_file
import general.path as path
import general.dataType as TYPE
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
from PIL import ImageTk, Image


class ManualTagClass(ttk.Frame):
    def __init__(self, master, buttonList, img_path, positions_path, output_position_path):
        ttk.Frame.__init__(self, master)
        self.close_flag = False
        self.output_position_path = output_position_path
        self.grid()
        self.winfo_toplevel().title("Label GUI-" + img_path.split('\\')[-1])
        self.winfo_toplevel().geometry("1200x900")
        self.img = cv2.imread(img_path)
        self.class_position = []
        self.nowPositionIndex = 0
        read_positions = read_file(positions_path, 'splitlines')
        self.positions = [position.split() for position in read_positions]
        self.len_position = len(self.positions)
        self.initWindow(buttonList)
        self.changeImage()

    def initWindow(self, buttonList):
        im = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)).resize(
            (800, 450), Image.ANTIALIAS)
        imgtk = ImageTk.PhotoImage(image=im)
        self.origin_img = tk.Label(self, image=imgtk)
        self.origin_img.image = imgtk
        self.origin_img.grid(row=1, column=0, rowspan=7, columnspan=2, padx=5)

        imgtk = ImageTk.PhotoImage(Image.open(
            'E:\\projects\\NTUST\\webimg-to-code\\datasetCode\\data_transform\\assest\\lay.jpg').resize((550, 360), Image.ANTIALIAS))
        lay = tk.Label(self, text="趙磊最棒", image=imgtk)
        lay.image = imgtk
        lay.grid(row=8, column=0, rowspan=7, columnspan=2, padx=5)
        self.label = tk.Label(self, text="趙磊最棒")
        self.now_position = tk.Label(self, text='{}/{}'.format(
            self.nowPositionIndex, self.len_position), font=25).grid(row=0, column=1, sticky=tk.W)
        self.closeButton = tk.Button(self, command=self.close_window, text="那得吧",
                                     width=10, pady=10, font=25).grid(row=0, column=0, sticky=tk.W)
        self.buttons = [self.create_btn(text, str(i))
                        for i, text in enumerate(buttonList)]
        for i, btn in enumerate(self.buttons):
            btn.grid(row=i+1, column=2, sticky=tk.W)

    def create_btn(self, text, tag):
        def cmd(): return self.assign_tag(tag)
        return tk.Button(self, command=cmd, text=text, width=20, pady=10, font=25)

    def changeImage(self):
        self.label.image = None
        print('position: ', self.nowPositionIndex,
              self.positions[self.nowPositionIndex])
        img = splitImage(self.img, self.positions[self.nowPositionIndex])
        print('subImg: ', img.shape)
        im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=im)
        self.label = tk.Label(self, image=imgtk)
        self.label.image = imgtk
        self.label.grid(row=0, column=2, sticky=tk.W)

    def assign_tag(self, tag):
        self.class_position.append(
            [tag]+self.positions[self.nowPositionIndex][1:])
        if self.nowPositionIndex+1 < self.len_position:
            self.nowPositionIndex += 1
            self.now_position = tk.Label(self, text='{}/{}'.format(
                self.nowPositionIndex, self.len_position), font=25).grid(row=0, column=1, sticky=tk.W)
            self.changeImage()
        else:
            showinfo("(*´Д`)つ))´∀`)", "心趙不宣，磊落不凡")
            write_file(self.class_position, self.output_position_path, 2)
            self.destroy()

    def close_window(self):
        self.close_flag = True
        self.destroy()

    def is_close(self):
        return self.close_flag


def splitImage(origin_image, position):
    (img_high, img_width, _) = origin_image.shape
    x, y = float(position[1])*img_width, float(position[2])*img_high
    w, h = float(position[3])*img_width, float(position[4])*img_high
    x, y, w, h = int(x), int(y), int(w), int(h)
    sub_img = origin_image[y:y+h, x: x+w]
    return sub_img


#  {button: 0, text: 1, title: 2}
def manual_class_tag(positions: list, origin_img, buttonList: list):
    class_position = []
    (img_high, img_width, _) = origin_img.shape
    interrupt = False

    # cv2.imshow("origin_img", origin_img)
    for i, position in enumerate(positions):
        x, y = float(position[1])*img_width, float(position[2])*img_high
        w, h = float(position[3])*img_width, float(position[4])*img_high
        x, y, w, h = int(x), int(y), int(w), int(h)
        sub_img = origin_img[y:y+h, x: x+w]

        cv2.imshow("sub_img", sub_img)
        key = cv2.waitKey()
        class_position.append([str(key-48)]+position[1:])
        print(key-48, "\n ------")
        if key-48 < 0 or key-48 > 9:
            cv2.destroyAllWindows()
            interrupt = True
            break

    return class_position, interrupt


def manual_class_tag_from_file(img_path, position_path):
    read_positions = read_file(position_path, 'splitlines')
    positions = [position.split() for position in read_positions]
    img = cv2.imread(img_path)
    class_position, interrupt = manual_class_tag(positions, img)
    return class_position, interrupt


def to_yolo_training_file(img_folder, positions_folder, data_length, target_path):
    print("img_folder: ", img_folder)
    print("positions_folder: ", positions_folder)
    print("target_path: ", target_path)
    with open(target_path, 'w+') as target:
        for index in range(data_length):
            img = cv2.imread(img_folder+str(index)+TYPE.IMG)
            (img_high, img_width, _) = img.shape
            boxs=[]
            read_positions = read_file(positions_folder+str(index)+TYPE.TXT, 'splitlines')
            positions = [position.split() for position in read_positions]
            for position in positions:
                min_x, min_y = float(position[1])*img_width, float(position[2])*img_high
                max_x, max_y = min_x + (float(position[3])*img_width), min_y + (float(position[4])*img_high)
                min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)
                box = ','.join([str(min_x), str(min_y), str(max_x), str(max_y), position[0]])
                boxs.append(box)
            
            line ="{} {}".format(img_folder+str(index)+TYPE.IMG, " ".join(boxs))   
            target.write(line+"\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = ManualTagClass(root, ['a', 'b'])
    root.mainloop()
