from Datas import Data
import wx


class frmSinifEkle(wx.MDIChildFrame):
    data = None

    def __init__(self, parent):
        wx.MDIChildFrame.__init__(self, parent, title='Sınıf Ekle', size=(350, 250))
        self.data = Data.Data()
        self.make_form()

    def make_form(self):
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        l1 = wx.StaticText(panel, -1, "sınıf İsmi")

        hbox1.Add(l1, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        self.t1 = wx.TextCtrl(panel)

        hbox1.Add(self.t1, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        vbox.Add(hbox1)

        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        l2 = wx.StaticText(panel, -1, "Klasör İsmi")

        hbox2.Add(l2, 1, wx.ALIGN_LEFT | wx.ALL, 5)
        self.t2 = wx.TextCtrl(panel)

        hbox2.Add(self.t2, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        vbox.Add(hbox2)

        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        self.btn = wx.Button(panel, -1, label='Ekle', size=(50, 20))
        hbox3.Add(self.btn, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        self.Bind(wx.EVT_BUTTON, self.OnAddSinif)
        vbox.Add(hbox3)

        panel.SetSizer(vbox)

    def OnAddSinif(self, event):
        self.data.add_data_sinif(self.t1.GetValue(), self.t2.GetValue())
        self.t1.Clear()
        self.t2.Clear()

