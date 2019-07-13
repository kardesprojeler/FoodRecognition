from Datas.Data import *
import wx
import wx.grid as grid

class frmSinifEkle(wx.Dialog):
    def __init__(self, parent):
        wx.Dialog.__init__(self, parent, title='Sınıf Ekle', size=(450, 500))
        self.current_row = -1
        self.make_form()

    def make_form(self):
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        l1 = wx.StaticText(panel, -1, "Sınıf İsmi")
        hbox1.Add(l1, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        self.tboxClass = wx.TextCtrl(panel)
        hbox1.Add(self.tboxClass, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)

        txtDoviz = wx.StaticText(panel, -1, "Döviz")
        hbox1.Add(txtDoviz, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        doviz_list = ["TL", "USD"]
        self.cbo_doviz = wx.ComboBox(panel, choices=doviz_list)
        hbox1.Add(self.cbo_doviz, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        vbox.Add(hbox1)

        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        txtKalasor = wx.StaticText(panel, -1, "Klasör İsmi")
        hbox2.Add(txtKalasor, 1, wx.ALIGN_LEFT | wx.ALL, 5)
        self.tboxKlasor = wx.TextCtrl(panel)
        hbox2.Add(self.tboxKlasor, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)

        txtFiyat = wx.StaticText(panel, -1, "Fiyat")
        hbox2.Add(txtFiyat, 1, wx.ALIGN_LEFT | wx.ALL, 5)
        self.tboxFiyat = wx.TextCtrl(panel)
        hbox2.Add(self.tboxFiyat, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)

        vbox.Add(hbox2)

        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        self.btnAdd = wx.Button(panel, -1, label='Ekle', size=(80, 30))
        hbox3.Add(self.btnAdd, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        self.btnAdd.Bind(wx.EVT_BUTTON, self.OnAddClass)
        self.btnAdd.SetBackgroundColour(wx.Colour(10, 200, 10))

        self.btnDelete = wx.Button(panel, -1, label='Sil', size=(80, 30))
        self.btnDelete.SetBackgroundColour(wx.Colour(192, 10, 10))
        hbox3.Add(self.btnDelete, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        self.btnDelete.Bind(wx.EVT_BUTTON, self.OnDeleteClass)

        self.btnSave = wx.Button(panel, -1, label='Kaydet', size=(80, 30))
        self.btnSave.SetBackgroundColour(wx.Colour(10, 90, 60))
        hbox3.Add(self.btnSave, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        self.btnSave.Bind(wx.EVT_BUTTON, self.onSave)

        vbox.Add(hbox3)

        #gridregion
        hbox4 = wx.BoxSizer(wx.HORIZONTAL)

        self.class_list = get_sinif_list()

        self.grid = wx.grid.Grid(panel, -1)
        self.grid.CreateGrid(0, 5)
        self.grid.SetColLabelValue(1, "Sınıf İsmi")
        self.grid.SetColLabelValue(2, "Klasör")
        self.grid.SetColLabelValue(3, "Fiyat")
        self.grid.SetColLabelValue(4, "Döviz")
        self.grid.SetColSize(1, 100)
        self.grid.HideCol(0)

        for i, class_ in enumerate(self.class_list):
            self.grid.AppendRows()
            self.grid.SetCellValue(self.grid.GetNumberRows() - 1, 0, str(class_.id))
            self.grid.SetCellValue(self.grid.GetNumberRows() - 1, 1, class_.sinifname)
            self.grid.SetCellValue(self.grid.GetNumberRows() - 1, 2, class_.foldername)
            self.grid.SetCellValue(self.grid.GetNumberRows() - 1, 3, str(class_.fiyat))
            self.grid.SetCellValue(self.grid.GetNumberRows() - 1, 4, class_.doviz)

        self.grid.Bind(wx.grid.EVT_GRID_SELECT_CELL, self.onGridRowChanGed)
        self.grid.SetSize(wx.Size(500, 600))
        hbox4.SetMinSize(wx.Size(500, 600))
        hbox4.Add(self.grid, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        vbox.Add(hbox4)
        #endregion

        panel.SetSizer(vbox)

    def OnAddClass(self, event):
        if self.tboxClass.GetValue() != "":
            self.grid.AppendRows()
            self.grid.SetCellValue(self.grid.GetNumberRows() - 1, 0, "-1")
            self.grid.SetCellValue(self.grid.GetNumberRows() - 1, 1, self.tboxClass.GetValue())
            self.grid.SetCellValue(self.grid.GetNumberRows() - 1, 2, self.tboxKlasor.GetValue())
            self.grid.SetCellValue(self.grid.GetNumberRows() - 1, 3, self.tboxFiyat.GetValue())
            self.grid.SetCellValue(self.grid.GetNumberRows() - 1, 4, self.cbo_doviz.GetValue())
            self.tboxClass.Clear()
            self.tboxKlasor.Clear()
            self.tboxFiyat.Clear()
            self.cbo_doviz.SetValue("")

        else:
            wx.MessageBox('Sınıf İsmi Boş Olamaz!', 'Attention', wx.OK | wx.ICON_WARNING)

    def OnDeleteClass(self, event):
        if self.current_row > -1 and self.grid.GetNumberRows() >= self.current_row + 1:
            class_id = self.grid.GetCellValue(self.current_row, 0)
            if class_id != "":
                self.grid.DeleteRows(self.current_row, updateLabels=True)
                self.grid.ForceRefresh()
                delete_class(int(class_id))

    def onGridRowChanGed(self, event):
        self.current_row = event.GetRow()

    def onSave(self, event):
        for i in range(self.grid.GetNumberRows()):

            update_class(int(self.grid.GetCellValue(i, 0)), self.grid.GetCellValue(i, 1), self.grid.GetCellValue(i, 2),
                         float(self.grid.GetCellValue(i, 3)), self.get_dovizref(self.grid.GetCellValue(i, 4)))


    def get_dovizref(self, dovizkod):
        if dovizkod == "TL":
            return 1
        elif dovizkod == "USD":
            return 2
        else:
            return -1
