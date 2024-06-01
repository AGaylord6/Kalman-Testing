'''
saving.py
Author: Andrew Gaylord

contains the saving functionality for kalman filter visualization
saves graphs to png and then embeds them in a pdf with contextual information

'''

import os
from matplotlib.backends.backend_pdf import PdfPages
import subprocess
from fpdf import FPDF
import numpy as np
import matplotlib.pyplot as plt


# declare global var for name out output directory to store plots in
outputDir = "plotOutput"

def saveFig(fig, fileName):
    '''
    saves fig to a png file in the outputDir directory with the name fileName
    also closes fig
    '''

    # absolute path to current directory
    my_path = os.path.dirname(os.path.abspath(__file__)) 
    saveDirectory = os.path.join(my_path, outputDir)

    fig.savefig(os.path.join(saveDirectory, fileName))

    plt.close(fig)



def savePDF(outputFile, pngDir):
    '''
    creates a report pdf using FPDF with all PNGs found in pngDir
    describes the graphs and their significance
    '''

    # absolute path to current directory
    my_path = os.path.dirname(os.path.abspath(__file__)) 
    pngDirectory = os.path.join(my_path, pngDir)

    # create the PDF object
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # iterate over all PNGs in the directory and add them to the pdf
    for i, png in enumerate(os.listdir(pngDirectory)):
        
        # add title and description
        pdf.cell(200, 10, txt="Plot " + str(i+1), ln=1, align='C')
        pdf.cell(200, 10, txt="This is a description of the graph and its significance", ln=1, align='L')

        # add the PNG to the pdf
        pdf.image(os.path.join(pngDirectory, png), x=10, y=pdf.get_y(), w=180)

        # add a page break if not the last PNG
        if i < len(os.listdir(pngDirectory))-1:
            pdf.add_page()

    # output the pdf to the outputFile
    pdf.output(outputFile)



def savePNGs(outputDir):
    '''
    saves all currently open plots as PNGs in outputDir and closes them
    '''

    # absolute path to current directory
    my_path = os.path.dirname(os.path.abspath(__file__)) 
    saveDirectory = os.path.join(my_path, outputDir)
    
    # get list of all figures
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]

    numPlots = 0
    # iterate over and save all plots tp saveDirectory
    for fig in figs:  
        numPlots += 1
        
        # save and close the current figure
        fig.savefig(os.path.join(saveDirectory, "plot" + str(numPlots) + ".png"))
        # fig.savefig(saveDirectory + "plot" + str(numPlots) + ".png")

        plt.close(fig)


def openFile(outputFile):
    # open the pdf file
    subprocess.Popen([outputFile], shell=True)

def oldPDF(outputFile):
    pass
    # Create the PdfPages object to which we will save the pages:
    # The with statement makes sure that the PdfPages object is closed properly at
    # the end of the block, even if an Exception occurs.
    # with PdfPages(outputFile) as pdf:

    #     fig_nums = plt.get_fignums()
    #     figs = [plt.figure(n) for n in fig_nums]
        
    #     # iterating over the numbers in list 
    #     for fig in figs:  
        
    #         # save and close the current figure
    #         fig.savefig(pdf, format='pdf') 
    #         pdf.attach_note("This is a note")
    #         plt.close(fig)
        
    #     # attach_note(self, text, positionRect=[-100, -100, 0, 0]) 
    #     # - Adds a new note to the page that will be saved next. 
    #     # The optional positionRect specifies the position on the page.

    #     # We can also set the file's metadata via the PdfPages object:
    #     d = pdf.infodict()
    #     d['Title'] = 'Kalman-Testing Output'
    #     d['Author'] = u'Jouni K. Sepp\xe4nen'
    #     d['Subject'] = 'Graphical output of Kalman-Testing simulation'
    #     d['Keywords'] = """IrishSat, UKF, Kalman Filter, CubeSat, Magnetometer, Gyroscope, Quaternion, Angular Velocity, 
    #                     Magnetic Field, Reaction Wheels, EOM, Unscented Kalman Filter, State Estimation, State Space, Measurement Space, 
    #                     Process Noise, Measurement Noise, Magnetic Field, Propagation, Simulation, Testing"""
