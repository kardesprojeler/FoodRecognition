from Datas import Data
import wx
from tkinter import messagebox

class frmTestImage(wx.MDIChildFrame):
    data = None

    def __init__(self, parent):
        wx.MDIChildFrame.__init__(self, parent, title='Test Resmi Ekle', size=(350, 250))
        self.data = Data.Data()
        self.make_form()

    def make_form(self):
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        l1 = wx.StaticText(panel, -1, "Sınıf İsmi")
        hbox1.Add(l1, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)

        self.siniflist = self.data.get_sinif_list()
        siniflar = []

        for sinif in self.siniflist:
            siniflar.append(sinif.sinifname)

        self.cbo_sinif = wx.ComboBox(panel, choices=siniflar)

        hbox1.Add(self.cbo_sinif, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        vbox.Add(hbox1)

        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        self.btn = wx.Button(panel, -1, label='Resim Ekle', size=(60, 20))
        hbox2.Add(self.btn, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        self.Bind(wx.EVT_BUTTON, self.OnAddImage)
        vbox.Add(hbox2)

        panel.SetSizer(vbox)

    def OnAddImage(self, event):
        if self.cbo_sinif.GetValue() == None or self.cbo_sinif.GetValue() == '':
            wx.MessageBox('Lütfen Sınıf Seçiniz', 'Attention', wx.OK | wx.ICON_WARNING)
            return
        label_number = self.siniflist[[s.sinifname for s in self.siniflist].index(
            self.cbo_sinif.GetValue()
        )].labelnumber
        self.data.add_test_image(self, label_number)