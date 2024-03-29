from Forms.frmSinifEkle import frmSinifEkle
from Datas.Data import *
import wx
from PIL import Image
import os

class frmPasteLabel(wx.Dialog):
    def __init__(self, parent, title, image, file_path, x_start, y_start, x_end, y_end):
        super(frmPasteLabel, self).__init__(parent, title=title, size=(300, 300),  style=wx.DEFAULT_FRAME_STYLE|wx.RESIZE_BORDER)
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
        self.image.Rescale(120, 120)
        imageCtrl = wx.StaticBitmap(panel, wx.ID_ANY, bitmap=self.image.ConvertToBitmap(3))
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
        with open(r'C:\Users\Durkan\Desktop\train_images.txt', 'a', encoding="utf-8") as file:
            file.write('\n' + self.file_path + ',' + str(x_start) + ',' + str(y_start) + ',' +
                       str(x_end) + ',' + str(y_end) + ',' + str(id))

        self.Destroy()


class frmImageShower(wx.MDIChildFrame):
    def __init__(self, parent):
        wx.MDIChildFrame.__init__(self, parent, title='Resimleri Göster',
                                  style=wx.VSCROLL|wx.HSCROLL|wx.DEFAULT_FRAME_STYLE)
        self.frame = wx.Frame(None, title='Photo Control')
        self.parent = parent
        self.panel = wx.Panel(self)
        self.moving = False
        self.click = False
        self.create_widgets()
        self.image_path_list_temp = []
        self.image_path_list_order_index = 0

    def show_image(self, file_path):
        img = wx.Image(file_path, wx.BITMAP_TYPE_ANY)
        img.Rescale(GeneralFlags.train_image_width.value, GeneralFlags.train_image_height.value)
        # scale the image, preserving the aspect ratio

        self.imageCtrl.SetBitmap(wx.Bitmap(img, wx.BITMAP_TYPE_ANY))
        self.photoTxt.SetValue(file_path)
        self.panel.Refresh()

    def create_widgets(self):
        img = wx.Image(GeneralFlags.train_image_width.value, GeneralFlags.train_image_height.value)
        self.imageCtrl = wx.StaticBitmap(self.panel, wx.ID_ANY, wx.Bitmap(img))

        self.imageCtrl.Bind(wx.EVT_LEFT_DOWN, self.onLeftDown)
        self.imageCtrl.Bind(wx.EVT_LEFT_UP, self.onLeftUp)
        self.imageCtrl.Bind(wx.EVT_MOTION, self.OnMove)

        self.photoTxt = wx.TextCtrl(self.panel, size=(200, -1))
        browseBtn = wx.Button(self.panel, label='Browse')
        browseBtn.Bind(wx.EVT_BUTTON, self.onBrowse)
        nextBtn = wx.Button(self.panel, label='Next')
        nextBtn.Bind(wx.EVT_BUTTON, self.onNext)
        addClassBtn = wx.Button(self.panel, label='Sınıf Ekle')
        addClassBtn.Bind(wx.EVT_BUTTON, self.onAddClass)

        self.mainSizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.mainSizer.Add(wx.StaticLine(self.panel, wx.ID_ANY),0, wx.ALL | wx.EXPAND, 5)
        self.mainSizer.Add(self.imageCtrl, 0, wx.ALL, 5)
        self.sizer.Add(self.photoTxt, 0, wx.ALL, 5)
        self.sizer.Add(browseBtn, 0, wx.ALL, 5)
        self.sizer.Add(nextBtn, 0, wx.ALL, 5)
        self.sizer.Add(addClassBtn, 0, wx.ALL, 5)

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
            self.image_path_list_temp = os.listdir(os.path.dirname(dialog.GetPath()))
            if os.path.dirname(dialog.GetPath()) != os.path.dirname(self.photoTxt.GetValue()):
                self.image_path_list_order_index = 0

        dialog.Destroy()

    def onView(self, file_path):
        self.show_image(file_path)

    def onLeftDown(self, event):
        self.click = True
        self.x_start, self.y_start = event.x, event.y
        return

    def onLeftUp(self, event):
        if self.moving and self.click:
            temp_images_path = r"C:\Users\Durkan\Documents\GitHub\FoodRecognition\images\temp_images\image.png"

            image = Image.open(self.photoTxt.GetValue())
            image = image.resize((GeneralFlags.train_image_width.value,
                                 GeneralFlags.train_image_height.value), Image.ANTIALIAS)
            image_crop = image.crop((self.x_start, self.y_start, event.x, event.y))

            image_crop.save(temp_images_path)
            image = wx.Image(temp_images_path)

            with frmPasteLabel(self.parent, "Etiketle", image,
                               self.photoTxt.GetValue(), self.x_start, self.y_start, event.x, event.y) as label_past:
                label_past.ShowModal()

        self.click = False
        self.moving = False
        return

    def OnMove(self, event):
        self.moving = True
        return

    def onNext(self, event):
        if len(self.image_path_list_temp) > 0:
            if self.image_path_list_order_index == len(self.image_path_list_temp) - 1:
                if wx.MessageBox('Bu klasördeki bütün dosyalar etiketlendi yeniden etiketlemek istermisiniz!',
                                 'Attention', wx.YES_NO | wx.ICON_WARNING) == wx.ID_YES:
                    self.image_path_list_order_index = 0

            file_path = self.image_path_list_temp[self.image_path_list_order_index]
            self.show_image(os.path.join(os.path.dirname(self.photoTxt.GetValue()), file_path))
            self.image_path_list_order_index += 1

    def onAddClass(self, event):
        frm = frmSinifEkle(self.parent)
        frm.Show(True)