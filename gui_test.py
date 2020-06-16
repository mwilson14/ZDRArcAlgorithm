from tkinter import *
from arcalg_framework_newforest import multi_case_algorithm_ML1_newforest
from arcalg_framework_realtime import multi_case_algorithm_ML1_realtime

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Here, we are creating our class, Window, and inheriting from the Frame
# class. Frame is a class from the tkinter module. (see Lib/tkinter/__init__)
class Window(Frame):

    # Define settings upon initialization. Here you can specify
    def __init__(self, master=None):
        
        # parameters that you want to send through the Frame class. 
        Frame.__init__(self, master)   

        #reference to the master widget, which is the tk window                 
        self.master = master

        #with that, we want to then run init_window, which doesn't yet exist
        self.init_window()

    #Creation of init_window
    def init_window(self):

        # changing the title of our master widget      
        self.master.title("ZDR Arc Algorithm")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        # creating a button instance
        quitButton = Button(self, text="Exit",command=self.client_exit)
        
        # creating a button instance
        printButton = Button(self, text="Run Archived Case",command=self.launch_archive)
        
        # creating a button instance
        realtimeButton = Button(self, text="Run Realtime",command=self.launch_realtime)

        # placing the button on my window
        quitButton.place(x=0, y=600)
        
        printButton.place(x=0, y=540)
        
        realtimeButton.place(x=0, y=570)

        #Add a label at the top of the algorithm
        L0 = Label(self, text = "Core Algorithm Settings")
        L0.pack()
        L0.place(x=150, y=5)
        
        #Add a label for storm tracking settings
        La = Label(self, text = "Tracking Algorithm Settings")
        La.pack()
        La.place(x=150, y=210)
        
        #Add a label for case date info
        Lb = Label(self, text = "Archived Case Info")
        Lb.pack()
        Lb.place(x=150, y=330)
        
        #Add entry for the FFD angle
        L = Label(self, text = "FFD Angle")
        L.pack()
        L.place(x=0, y=30)
        
        #Add a data entry window
        self.entrythingy = Entry()
        self.entrythingy.pack()
        self.entrythingy.place(x=150, y=30)

        # here is the application variable
        self.contents = StringVar()
        # set it to some value
        self.contents.set(" ")
        # tell the entry widget to watch this variable
        self.entrythingy["textvariable"] = self.contents
        
        #Add entry for the ZDR threshold
        L1 = Label(self, text = "ZDR Threshold (dB)")
        L1.pack()
        L1.place(x=0, y=60)
        
        #Add a data entry window
        self.entrythingy1 = Entry()
        self.entrythingy1.pack()
        self.entrythingy1.place(x=150, y=60)

        # here is the application variable
        self.contents1 = StringVar()
        # set it to some value
        self.contents1.set("3.25")
        # tell the entry widget to watch this variable
        self.entrythingy1["textvariable"] = self.contents1
        
        #Add entry for the KDP threshold
        L2 = Label(self, text = "KDP Threshold (deg/km)")
        L2.pack()
        L2.place(x=0, y=90)
        
        #Add a data entry window
        self.entrythingy2 = Entry()
        self.entrythingy2.pack()
        self.entrythingy2.place(x=150, y=90)

        # here is the application variable
        self.contents2 = StringVar()
        # set it to some value
        self.contents2.set("1.5")
        # tell the entry widget to watch this variable
        self.entrythingy2["textvariable"] = self.contents2
        
        #Add entry for the first tracking threshold
        L3 = Label(self, text = "Z Threshold 1")
        L3.pack()
        L3.place(x=0, y=240)
        
        #Add a data entry window
        self.entrythingy3 = Entry()
        self.entrythingy3.pack()
        self.entrythingy3.place(x=150, y=240)

        # here is the application variable
        self.contents3 = StringVar()
        # set it to some value
        self.contents3.set("45")
        # tell the entry widget to watch this variable
        self.entrythingy3["textvariable"] = self.contents3
        
        #Add entry for the second tracking threshold
        L4 = Label(self, text = "Z Threshold 2")
        L4.pack()
        L4.place(x=0, y=270)
        
        #Add a data entry window
        self.entrythingy4 = Entry()
        self.entrythingy4.pack()
        self.entrythingy4.place(x=150, y=270)

        # here is the application variable
        self.contents4 = StringVar()
        # set it to some value
        self.contents4.set("50")
        # tell the entry widget to watch this variable
        self.entrythingy4["textvariable"] = self.contents4
        
        #Add entry for big storm area
        L5 = Label(self, text = "Big Storm Area (sq. km)")
        L5.pack()
        L5.place(x=0, y=300)
        
        #Add a data entry window
        self.entrythingy5 = Entry()
        self.entrythingy5.pack()
        self.entrythingy5.place(x=150, y=300)

        # here is the application variable
        self.contents5 = StringVar()
        # set it to some value
        self.contents5.set("300")
        # tell the entry widget to watch this variable
        self.entrythingy5["textvariable"] = self.contents5
        
        #Add entry for case year
        L6 = Label(self, text = "Year")
        L6.pack()
        L6.place(x=0, y=360)
        
        #Add a data entry window
        self.entrythingy6 = Entry()
        self.entrythingy6.pack()
        self.entrythingy6.place(x=150, y=360)

        # here is the application variable
        self.contents6 = StringVar()
        # set it to some value
        self.contents6.set("2016")
        # tell the entry widget to watch this variable
        self.entrythingy6["textvariable"] = self.contents6
        
        #Add entry for case month
        L7 = Label(self, text = "Month")
        L7.pack()
        L7.place(x=0, y=390)
        
        #Add a data entry window
        self.entrythingy7 = Entry()
        self.entrythingy7.pack()
        self.entrythingy7.place(x=150, y=390)

        # here is the application variable
        self.contents7 = StringVar()
        # set it to some value
        self.contents7.set("3")
        # tell the entry widget to watch this variable
        self.entrythingy7["textvariable"] = self.contents7
        
         #Add entry for case day
        L8 = Label(self, text = "Day")
        L8.pack()
        L8.place(x=0, y=420)
        
        #Add a data entry window
        self.entrythingy8 = Entry()
        self.entrythingy8.pack()
        self.entrythingy8.place(x=150, y=420)

        # here is the application variable
        self.contents8 = StringVar()
        # set it to some value
        self.contents8.set("30")
        # tell the entry widget to watch this variable
        self.entrythingy8["textvariable"] = self.contents8
        
         #Add entry for case start hour
        L9 = Label(self, text = "Start Hour (UTC)")
        L9.pack()
        L9.place(x=0, y=450)
        
        #Add a data entry window
        self.entrythingy9 = Entry()
        self.entrythingy9.pack()
        self.entrythingy9.place(x=150, y=450)

        # here is the application variable
        self.contents9 = StringVar()
        # set it to some value
        self.contents9.set("21")
        # tell the entry widget to watch this variable
        self.entrythingy9["textvariable"] = self.contents9
        
         #Add entry for case start minute
        L10 = Label(self, text = "Start Minute (UTC)")
        L10.pack()
        L10.place(x=0, y=480)
        
        #Add a data entry window
        self.entrythingy10 = Entry()
        self.entrythingy10.pack()
        self.entrythingy10.place(x=150, y=480)

        # here is the application variable
        self.contents10 = StringVar()
        # set it to some value
        self.contents10.set("0")
        # tell the entry widget to watch this variable
        self.entrythingy10["textvariable"] = self.contents10
        
          #Add entry for case duration
        L11 = Label(self, text = "Case Duration (hours)")
        L11.pack()
        L11.place(x=0, y=510)
        
        #Add a data entry window
        self.entrythingy11 = Entry()
        self.entrythingy11.pack()
        self.entrythingy11.place(x=150, y=510)

        # here is the application variable
        self.contents11 = StringVar()
        # set it to some value
        self.contents11.set("1.5")
        # tell the entry widget to watch this variable
        self.entrythingy11["textvariable"] = self.contents11

          #Add entry for ZDR Calibration
        L12 = Label(self, text = "ZDR Calibration (dB)")
        L12.pack()
        L12.place(x=0, y=120)
        
        #Add a data entry window
        self.entrythingy12 = Entry()
        self.entrythingy12.pack()
        self.entrythingy12.place(x=150, y=120)

        # here is the application variable
        self.contents12 = StringVar()
        # set it to some value
        self.contents12.set("0.0")
        # tell the entry widget to watch this variable
        self.entrythingy12["textvariable"] = self.contents12
        
          #Add entry for radar site
        L13 = Label(self, text = "Radar Site")
        L13.pack()
        L13.place(x=0, y=150)
        
        #Add a data entry window
        self.entrythingy13 = Entry()
        self.entrythingy13.pack()
        self.entrythingy13.place(x=150, y=150)

        # here is the application variable
        self.contents13 = StringVar()
        # set it to some value
        self.contents13.set("")
        # tell the entry widget to watch this variable
        self.entrythingy13["textvariable"] = self.contents13
        
          #Add entry for storm motion
        L14 = Label(self, text = "Storm Motion (deg)")
        L14.pack()
        L14.place(x=0, y=180)
        
        #Add a data entry window
        self.entrythingy14 = Entry()
        self.entrythingy14.pack()
        self.entrythingy14.place(x=150, y=180)

        # here is the application variable
        self.contents14 = StringVar()
        # set it to some value
        self.contents14.set("")
        # tell the entry widget to watch this variable
        self.entrythingy14["textvariable"] = self.contents14

    def client_exit(self):
        exit()
        
    def print_stuff(self):
        data = float(self.contents.get()) 
        print(data, 'FFD angle (degrees)')
        data1 = float(self.contents1.get())
        print(data1, 'ZDR Threshold (dB)')
        data2 = float(self.contents2.get())
        print(data2, 'KDP Threshold (deg/km)')
        data3 = float(self.contents3.get())
        print(data3, 'Z Threshold 1 (dBZ)')
        data4 = float(self.contents4.get())
        print(data4, 'Z Threshold 2 (dBZ)')
        data5 = float(self.contents5.get())
        print(data5, 'Big Storm Area (sq. km)')
        data6 = int(self.contents6.get())
        print(data6, 'Year')
        data7 = int(self.contents7.get())
        print(data7, 'Month')
        data8 = int(self.contents8.get())
        print(data8, 'Day')
        data9 = int(self.contents9.get())
        print(data9, 'Start Hour (UTC)')
        data10 = int(self.contents10.get())
        print(data10, 'Start Minute (UTC)')
        data11 = float(self.contents11.get())
        print(data11, 'Case Duration (hours)')
        data12 = float(self.contents12.get())
        print(data12, 'ZDR Calibration')
        data13 = str(self.contents13.get())
        print(data13, 'Radar Site')
        data14 = str(self.contents14.get())
        print(data14, 'Storm Motion (deg)')
        
    def launch_archive(self):
        ffdangle = float(self.contents.get()) 
        print(ffdangle, 'FFD angle (degrees)')
        zdr_t = float(self.contents1.get())
        print(zdr_t, 'ZDR Threshold (dB)')
        kdp_t = float(self.contents2.get())
        print(kdp_t, 'KDP Threshold (deg/km)')
        z1 = float(self.contents3.get())
        print(z1, 'Z Threshold 1 (dBZ)')
        z2 = float(self.contents4.get())
        print(z2, 'Z Threshold 2 (dBZ)')
        bs1 = float(self.contents5.get())
        print(bs1, 'Big Storm Area (sq. km)')
        year1 = int(self.contents6.get())
        print(year1, 'Year')
        month1 = int(self.contents7.get())
        print(month1, 'Month')
        day1 = int(self.contents8.get())
        print(day1, 'Day')
        hour1 = int(self.contents9.get())
        print(hour1, 'Start Hour (UTC)')
        min1 = int(self.contents10.get())
        print(min1, 'Start Minute (UTC)')
        dur1 = float(self.contents11.get())
        print(dur1, 'Case Duration (hours)')
        zdr_cal = float(self.contents12.get())
        print(zdr_cal, 'ZDR Calibration')
        site = (self.contents13.get())
        print(site, 'Radar Site')
        storm_motion = float(self.contents14.get())
        print(storm_motion, 'Storm Motion (deg)')
        
        print(" ")
        print('Running Algorithm')

        tracks_dataframe, zdroutlines = multi_case_algorithm_ML1_newforest(ffdangle,zdr_t,kdp_t,z1,z2,bs1,70,2,year1,month1,day1,hour1,min1,dur1,zdr_cal,site,storm_motion, track_dis=10)
        
        tracks_dataframe.to_pickle('ARCDEV_GUI'+str(year1)+str(month1)+str(day1)+str(site)+'.pkl')


    def launch_realtime(self):
        ffdangle = float(self.contents.get()) 
        print(ffdangle, 'FFD angle (degrees)')
        zdr_t = float(self.contents1.get())
        print(zdr_t, 'ZDR Threshold (dB)')
        kdp_t = float(self.contents2.get())
        print(kdp_t, 'KDP Threshold (deg/km)')
        z1 = float(self.contents3.get())
        print(z1, 'Z Threshold 1 (dBZ)')
        z2 = float(self.contents4.get())
        print(z2, 'Z Threshold 2 (dBZ)')
        bs1 = float(self.contents5.get())
        print(bs1, 'Big Storm Area (sq. km)')
        year1 = int(self.contents6.get())
        print(year1, 'Year')
        month1 = int(self.contents7.get())
        print(month1, 'Month')
        day1 = int(self.contents8.get())
        print(day1, 'Day')
        hour1 = int(self.contents9.get())
        print(hour1, 'Start Hour (UTC)')
        min1 = int(self.contents10.get())
        print(min1, 'Start Minute (UTC)')
        dur1 = float(self.contents11.get())
        print(dur1, 'Case Duration (hours)')
        zdr_cal = float(self.contents12.get())
        print(zdr_cal, 'ZDR Calibration')
        site = (self.contents13.get())
        print(site, 'Radar Site')
        storm_motion = float(self.contents14.get())
        print(storm_motion, 'Storm Motion (deg)')

        print(" ")
        print('Running Algorithm')

        tracks_dataframe, zdroutlines = multi_case_algorithm_ML1_realtime(ffdangle,zdr_t,kdp_t,z1,z2,bs1,2,
                                                                zdr_cal,station=site, Bunkers_m=storm_motion, track_dis=10)

        tracks_dataframe.to_pickle('ARCDEV_GUI'+str(year1)+str(month1)+str(day1)+str(site)+'.pkl')

        
# root window created. Here, that would be the only window, but
# you can later have windows within windows.
root = Tk()

root.geometry("400x650")

#creation of an instance
app = Window(root)

#mainloop 
root.mainloop()  
