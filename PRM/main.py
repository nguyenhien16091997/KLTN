import PySimpleGUI as sg
from Uploadfile import handleMissingValue, caseForClassification, CaculateMissingValue
import numpy as np
import os

if __name__ == '__main__':
    #input file
    layout1 = [[sg.Text('Chọn file mà bạn cần phân tích:')],
               [sg.Text('Đường dẫn file: ')],
               [sg.Input(), sg.FileBrowse()],
               [sg.Checkbox('Xóa dòng bị thiếu dữ liệu', default=True)],
               [sg.Checkbox('Thay thế giá trị bị thiếu bằng số 0 ', default=False)],
               [sg.Checkbox('Thay thế giá trị bị thiếu giá trị xuất hiện nhiều nhất ', default=False)],
               [sg.Checkbox('Thay thế giá trị bị thiếu bằng giá trị gần nhất ', default=False)],
               # [sg.Text('Dữ liệu bị thiếu giá trị :'), sg.ReadButton('Kiểm tra', button_color=('white', 'springgreen4'), key='PcMvalue'), sg.Text('%')],
               [sg.Text('Phần trăm thiếu giá trị cho dữ liệu'), sg.InputCombo(['5', '10', '20', '30'], readonly=True), sg.Text('%')],
               [sg.OK(), sg.Cancel()]]
    window1 = sg.Window('PRM',layout1)

    temp_Submit = False
    while True:
        button, values = window1.Read()
        if button is None or button == 'Cancel':
            window1.Close()
            break
        elif button == 'PcMvalue':
            if values[0] == '':
                sg.Popup('Vui lòng chọn file mà bạn cần phân tích !')
            else:
                txt = CaculateMissingValue(values[0])
                window1.Element('PcMvalue').Update(txt)
        else:
            if values[0] == '':
                sg.Popup('Vui lòng chọn file mà bạn cần phân tích !')
            else:
                chose = []
                for i in range(5):
                    if values[i] == True:
                        chose.append(i)

                #read resuilt and push to table

                list = handleMissingValue(values[0], chose, values[5])

                # avg
                avg1 = ''
                avg2 = ''
                avg3 = ''
                avg4 = ''
                avg5 = ''
                if 1 in chose:
                    avg1 = round(np.average(list[0].values[:, 3].astype(float)), 2)
                if 2 in chose:
                    avg2 = round(np.average(list[1].values[:, 3].astype(float)), 2)
                if 3 in chose:
                    avg3 = round(np.average(list[2].values[:, 3].astype(float)), 2)
                if 4 in chose:
                    avg4 = round(np.average(list[3].values[:, 3].astype(float)), 2)
                avg5 = round(np.average(list[17].values[:, 3].astype(float)), 2)
                #Name File
                nameFile = os.path.splitext(os.path.basename(values[0]))[0]

                if values[5] == '0':
                    values[5] = CaculateMissingValue(values[0]).astype('str')
                #show result
                layout2 = [
                            [sg.ReadButton('PHÂN LỚP', button_color=('white', 'springgreen4'), key='Classification'), sg.Text('Dữ bị thiếu: '+values[5]+'%'), sg.Text('Tên file: '+nameFile)],
                            [sg.Text('Xóa dòng có khoảng trắng: ', size=(62, 1)), sg.Text('Thay thế bằng giá trị xuất hiện nhiều nhất cùng thuộc tính: ')],
                            [sg.Table(values=list[0].values.tolist(), headings=list[0].columns.values.tolist(), display_row_numbers=True,
                                auto_size_columns=False, num_rows=7),
                                sg.Table(values=list[1].values.tolist(), headings=list[1].columns.values.tolist(), display_row_numbers=True,
                                auto_size_columns=False, num_rows=7)],
                            [sg.Text('Time: {} second'.format(list[4])), sg.Text('Ram: {}%'.format(list[9])), sg.Text('CPU: {}%'.format(list[13])), sg.Text('DKL: {}%'.format(list[21])), sg.Text('DCX: {}%'.format(list[25]), size=(20, 1)),
                             sg.Text('Time: {} second'.format(list[5])), sg.Text('Ram: {}%'.format(list[10])), sg.Text('CPU: {}%'.format(list[14])), sg.Text('DKL: {}%'.format(list[22])), sg.Text('DCX: {}%'.format(list[26]))],
                            [sg.Text('Thay thế bằng số 0: ', size=(62, 1)), sg.Text('Thay thế bằng giá trị lân cận: ')],
                            [sg.Table(values=list[2].values.tolist(), headings=list[2].columns.values.tolist(), display_row_numbers=True,
                                 auto_size_columns=False, num_rows=7),
                                sg.Table(values=list[3].values.tolist(), headings=list[3].columns.values.tolist(), display_row_numbers=True,
                                 auto_size_columns=False, num_rows=7)],
                            [sg.Text('Time: {} second'.format(list[6])), sg.Text('Ram: {}%'.format(list[11])), sg.Text('CPU: {}%'.format(list[15])), sg.Text('DKL: {}%'.format(list[23])), sg.Text('DCX: {}%'.format(list[27]), size=(20, 1)),
                             sg.Text('Time: {} second'.format(list[7])), sg.Text('Ram: {}%'.format(list[12])), sg.Text('CPU: {}%'.format(list[16])), sg.Text('DKL: {}%'.format(list[24])), sg.Text('DCX: {}%'.format(list[28]))],
                            [sg.Text('Tập train: ', size=(62, 1))],
                            [sg.Table(values=list[17].values.tolist(), headings=list[17].columns.values.tolist(), display_row_numbers=True,
                                 auto_size_columns=False, num_rows=7)],
                            [sg.Text('Time: {} second'.format(list[18])), sg.Text('Ram: {}%'.format(list[19])), sg.Text('CPU: {}%'.format(list[20])), sg.Text('DCX: {}%'.format(list[29]))]

                ]
                #Resuilt after handle with PRM
                window2 = sg.Window('PRM', grab_anywhere=False).Layout(layout2)
                # window1.Element('St').Update('')
                while True:
                    sg.PopupAnimated(None)  # close all Animated Popups
                    event, values = window2.Read()
                    # event Classification
                    if event is None:
                        window2.Close()
                        break
                    elif event == 'Classification':
                        itemSet = {}
                        # Layout for Classification
                        column = list[8].columns.values.tolist()
                        column = column[:-1]  # delete outcome
                        layout3 = [
                            [sg.Text('Vui lòng chọn giá trị bạn cần phân tích !')]
                        ]
                        for i in column:
                            # unique array
                            x = np.array(list[8].loc[:, i].values.tolist())
                            x = np.unique(x)
                            x = x.tolist()

                            layout3.append(
                                [sg.Text(i + " :", size=(15, 1)),
                                 sg.InputCombo(x, enable_events=True, key=i, size=(15, 1))])

                        layout3.append([sg.Submit('Xong'), sg.Button('Hủy')])
                        window3 = sg.Window('PHÂN LỚP').Layout(layout3)
                        while True:
                            event, values = window3.Read()
                            if event is None or event == 'Hủy':  # always check for closed window
                                window3.Close()
                                break
                            elif event == 'Xong' and len(itemSet) != 0:
                                nameRs = ['Loại dữ liệu','Lớp', 'Độ chính xác(%)', 'Độ tin cậy(%)']
                                lrs = caseForClassification(list, itemSet)
                                layout4 = [[sg.Text('Loại dữ liệu', size=(50, 1)), sg.Text('Lớp', size=(10, 1)), sg.Text('Độ chính xác (%)', size=(15, 1)), sg.Text('Độ tin cậy (%)', size=(15, 1))],
                                           [sg.Text('Dữ liệu gốc', size=(50, 1)), sg.Text(lrs[0][0], size=(10, 1)), sg.Text(lrs[0][1], size=(15, 1)), sg.Text(lrs[0][2], size=(15, 1))],
                                           [sg.Text('Xóa dòng bị thiếu dữ liệu', size=(50, 1)), sg.Text(lrs[1][0], size=(10, 1)), sg.Text(lrs[1][1], size=(15, 1)), sg.Text(lrs[1][2], size=(15, 1))],
                                           [sg.Text('Thay thế giá trị bị thiếu bằng số 0', size=(50, 1)), sg.Text(lrs[2][0], size=(10, 1)), sg.Text(lrs[2][1], size=(15, 1)), sg.Text(lrs[2][2], size=(15, 1))],
                                           [sg.Text('Thay thế giá trị bị thiếu giá trị xuất hiện nhiều nhất', size=(50, 1)), sg.Text(lrs[3][0], size=(10, 1)), sg.Text(lrs[3][1], size=(15, 1)), sg.Text(lrs[3][2], size=(15, 1))],
                                           [sg.Text('Thay thế giá trị bị thiếu bằng giá trị gần nhất', size=(50, 1)), sg.Text(lrs[4][0], size=(10, 1)), sg.Text(lrs[4][1], size=(15, 1)), sg.Text(lrs[4][2], size=(15, 1))]]

                                window4 = sg.Window('KẾT QUẢ').Layout(layout4)
                                window4.Read()
                            else:
                                for i in column:
                                    window3.Element(i).Update(list[8].loc[:, i].values.tolist())
                                    itemSet[i] = values[i]