from fileinput import filename

from Datas.Data import *
import wx
from PIL import Image
import os

class frmPasteLabel(wx.Dialog):
    def __init__(self, parent, title, image, file_path, x_start, y_start, x_end, y_end):
        super(frmPasteLabel, self).__init__(parent, title=title, size=(500, 500))
        self.image = image
        self.make_form()
        self.file_path = file_path
        self.x_start = x_start
        self.y_start = y_start
        self.x_end = x_end
        self.y_end = y_end

    def make_form(self):
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        l1 = wx.StaticText(panel, -1, "Sınıf İsmi")
        hbox1.Add(l1, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)

        self.siniflist = get_sinif_list()
        siniflar = []

        for sinif in self.siniflist:
            siniflar.append(sinif.sinifname)

        self.cbo_sinif = wx.ComboBox(panel, choices=siniflar)

        hbox1.Add(self.cbo_sinif, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        vbox.Add(hbox1)

        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        imageCtrl = wx.StaticBitmap(panel, wx.ID_ANY, bitmap=self.image)
        hbox2.Add(imageCtrl, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        vbox.Add(hbox2)

        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        self.btn = wx.Button(panel, -1, label='Ekle', size=(60, 20))
        hbox3.Add(self.btn, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        self.Bind(wx.EVT_BUTTON, self.on_add_label)
        vbox.Add(hbox3)

        panel.SetSizer(vbox)

    def on_add_label(self, event):
        if self.cbo_sinif.GetValue() == None or self.cbo_sinif.GetValue() == '':
            wx.MessageBox('Lütfen Sınıf Seçiniz', 'Attention', wx.OK | wx.ICON_WARNING)
            return
        label_number = self.siniflist[[s.sinifname for s in self.siniflist].index(
            self.cbo_sinif.GetValue()
        )].id
        self.add_label(label_number, self.x_start, self.y_start, self.x_end, self.y_end)

    def add_label(self, id, x_start, y_start, x_end, y_end):
        with open(r'C:\Users\BULUT\Desktop\train_datas.txt', 'a', encoding="utf-8") as file:
            file.write('\n' + self.file_path + ',' + str(x_start) + ',' + str(y_start) + ',' +
                       str(x_end) + ',' + str(y_end) + ',' + str(id))

        self.Destroy()


class frmImageShower(wx.MDIChildFrame):
    def __init__(self, parent):
        wx.MDIChildFrame.__init__(self, parent, title='Resimleri Göster', size=(350, 300))
        self.frame = wx.Frame(None, title='Photo Control')
        self.parent = parent
        self.panel = wx.Panel(self)
        self.PhotoMaxSize = 240
        self.moving = False
        self.click = False
        self.create_widgets()

    def show_image(self, file_path):
        img = wx.Image(file_path, wx.BITMAP_TYPE_ANY)
        # scale the image, preserving the aspect ratio

        self.imageCtrl.SetBitmap(wx.Bitmap(img, wx.BITMAP_TYPE_ANY))
        self.photoTxt.SetValue(file_path)
        self.panel.Refresh()

    def create_widgets(self):
        instructions = 'Browse for an image'
        img = wx.Image(240, 240)
        self.imageCtrl = wx.StaticBitmap(self.panel, wx.ID_ANY, wx.Bitmap(img))

        self.imageCtrl.Bind(wx.EVT_LEFT_DOWN, self.onLeftDown)
        self.imageCtrl.Bind(wx.EVT_LEFT_UP, self.onLeftUp)
        self.imageCtrl.Bind(wx.EVT_MOTION, self.OnMove)

        instructLbl = wx.StaticText(self.panel, label=instructions)
        self.photoTxt = wx.TextCtrl(self.panel, size=(200, -1))
        browseBtn = wx.Button(self.panel, label='Browse')
        browseBtn.Bind(wx.EVT_BUTTON, self.onBrowse)
        nextBtn = wx.Button(self.panel, label='Next')
        nextBtn.Bind(wx.EVT_BUTTON, self.onNext)

        self.mainSizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.mainSizer.Add(wx.StaticLine(self.panel, wx.ID_ANY),0, wx.ALL | wx.EXPAND, 5)
        self.mainSizer.Add(instructLbl, 0, wx.ALL, 5)
        self.mainSizer.Add(self.imageCtrl, 0, wx.ALL, 5)
        self.sizer.Add(self.photoTxt, 0, wx.ALL, 5)
        self.sizer.Add(browseBtn, 0, wx.ALL, 5)
        self.sizer.Add(nextBtn, 0, wx.ALL, 5)

        self.mainSizer.Add(self.sizer, 0, wx.ALL, 5)

        self.panel.SetSizer(self.mainSizer)
        self.mainSizer.Fit(self)

        self.panel.Layout()

    def onBrowse(self, event):
        """
        Browse for file
        """
        wildcard = "JPEG files (*.jpg)|*.jpg"
        dialog = wx.FileDialog(None, "Choose a file",
                               wildcard=wildcard,
                               style=wx.FD_OPEN)

        if dialog.ShowModal() == wx.ID_OK:
            self.onView(dialog.GetPath())
        dialog.Destroy()

    def onView(self, file_path):
        self.show_image(file_path)

    def onLeftDown(self, event):
        self.click = True
        self.x_start, self.y_start = event.x, event.y
        return

    def onLeftUp(self, event):
        if self.moving and self.click:
            temp_images_path = r"C:\Users\BULUT\Documents\GitHub\FoodRecognition\images\temp_images\image.png"
            image_crop = Image.open(self.photoTxt.GetValue()).crop((self.x_start, self.y_start, event.x, event.y))
            image_crop.save(temp_images_path)
            image = wx.Image(temp_images_path)

            with frmPasteLabel(self.parent, "Etiketle", image.ConvertToBitmap(3),
                               self.photoTxt.GetValue(), self.x_start, self.y_start, event.x, event.y) as label_past:
                label_past.ShowModal()

        self.click = False
        self.moving = False
        return

    def OnMove(self, event):
        self.moving = True
        return

    def onNext(self, event):
        if self.photoTxt.GetValue() != "":
            dir_name = os.path.dirname(self.photoTxt.GetValue())
            file_array = os.listdir(dir_name)

            with open(r'C:\Users\BULUT\Desktop\train_datas.txt', encoding='utf-8') as file:
                content = file.read()
            for file_name in file_array:
                if file_name.endswith(".jpg") or file_name.endswith(".png") or file_name.endswith(".gif"):
                    file_path = os.path.join(dir_name, file_name)
                    show = True
                    for line in content.split('\n'):
                        dir = line.split(',')[0]
                        if dir == file_path:
                            show = False
                    if show:
                        self.show_image(file_path)
                        break
