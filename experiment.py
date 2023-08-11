import os
import traceback
from pathlib import Path

import pandas as pd
from osgeo import gdal
from helpers import *
from imputation import Imputation as imp
import numpy as np



class Imputation:
    def __init__(self, inputFile, scaleFactor=0.00002, outputFile=None):
        self.inputFilePath = inputFile
        self.outputFilePath = outputFile
        self.scaleFactor = scaleFactor
        self.imputedArrays = []
        self.fileName = Path(self.inputFilePath).stem
        # print("filename",self.fileName)
        self.percent = None
        self.algos = ['siLRTC', 'haLRTC', 'CMTF', 'CMSI', 'cpALS', 'Base']
        try:
            self.data = gdal.Open(self.inputFilePath)
            self.geoTrans = self.data.GetGeoTransform()
            self.projection = self.data.GetProjection()
            self.rasterArray = self.data.ReadAsArray().T * scaleFactor
            self.imgWidth = self.rasterArray.shape[0]
            self.imgHeight = self.rasterArray.shape[1]
            if self.imgHeight != self.imgWidth:
                croppedShape  = min(self.imgWidth, self.imgHeight)
                self.rasterArray = self.rasterArray[:croppedShape,:croppedShape,:]
            # print(self.rasterArray.shape)
            self.impute = imp(self.rasterArray)
            # self.impute.show_img(self.impute.img[:, :, :3])


        except:
            print("Unable to read input file using gdal")
            print("Error: ", traceback.print_exc())

    def createMissingPixels(self, percent=10, method='random'):
        self.percent = percent
        self.mask, self.corruptImage = self.impute.synthetic_mask((100 - self.percent)/100)
        # print(self.mask,self.corruptImage)
        print("Corrupt Image: ", self.corruptImage.shape)
        path = str(self.outputFilePath + '/' + str(self.fileName))
        checkFolder = os.path.isdir(path)
        if not checkFolder:
            os.makedirs(path)
            print('created Folder', path)
        self.CreateGeoTiff(outRaster= str(path + '/' + str(self.percent) + 'missingNew.tif'), data=self.corruptImage,
                           geo_transform=self.geoTrans,projection=self.projection)

        # self.impute.show_img(self.corruptImage[:,:,:3])

    def predictMissingPixels(self, outputFolder, algo= None, tensorRank = 12):
        if algo is None:
            print("Running on all algorithms")
            self.results = {}
            if self.impute.get_corrupt_img() is None:
                self.impute.generate_mask(approximate_mask=True)
                print("Generating Approximate Mask")
            self.imputedArrays, self.algoRuntime = self.impute.impute(speed="slow", hil=0, skip_vis=0)
            # print(len(self.imputedArrays),self.algoRuntime)
            self.evalResults = []
            self.runTime = []

            for i in range(len(self.algos)- 1):
                path = str(outputFolder + self.algos[i] + '/' +str(filename))
                # print("outputPath = ",path)
                checkFolder = os.path.isdir(path)
                if not checkFolder:
                    os.makedirs(path)
                    print('created Folder', path)
                self.CreateGeoTiff(outRaster=str(path+'/'+str(self.percent)+self.algos[i]+'New.tif'),
                                   data=self.imputedArrays[i],
                                   geo_transform=self.geoTrans, projection=self.projection)
                self.evalResults.append(RSE(self.rasterArray,self.imputedArrays[i]))
                self.runTime.append(self.algoRuntime[i+1] - self.algoRuntime[i])
            self.runTime.append(None)
            self.evalResults.append(RSE(self.rasterArray,self.corruptImage))
            self.results = {'fileName': [self.fileName] * len(self.algos),
                             'missingPercent': [str(self.percent)] * len(self.algos),
                             'algo': self.algos, 'RSE': self.evalResults, 'runTime': self.runTime}
            # print(self.tempDict)
            return self.results


    def CreateGeoTiff(self,outRaster, data, geo_transform, projection):
        data = data.T
        driver = gdal.GetDriverByName('ENVI')

        no_bands, rows, cols = data.shape
        # print(data.shape)
        DataSet = driver.Create(outRaster, cols, rows, no_bands, gdal.GDT_Float32)
        DataSet.SetGeoTransform(geo_transform)
        DataSet.SetProjection(projection)
        for i, image in enumerate(data, 1):
            # print(i,image)
            DataSet.GetRasterBand(i).WriteArray(image)
        DataSet.FlushCache()
        DataSet = None


# path = "/Users/bunny/PycharmProjects/Imputation/data_/MI/dataset2/"

path = "/Users/bunny/PycharmProjects/Imputation/data_/chandrayan/dataset2/"

if __name__ == "__main__":
    finalDataframe = pd.DataFrame(columns=['fileName', 'missingPercent', 'algo', 'RSE', 'runTime'])
    for filename in os.listdir(path):
        if filename.endswith('.IMG'):
            print(filename)
            for missingPercent in range(10, 60, 10):
                filePath = os.path.join(path, filename)
                # print(filePath)
                impute = Imputation(inputFile=filePath, outputFile=path)
                impute.createMissingPixels(percent=missingPercent)
                results = impute.predictMissingPixels(outputFolder=path)
                # print(results)
                finalDataframe = pd.concat([finalDataframe,pd.DataFrame(results)],ignore_index=True)
                # print(finalDataframe)
            #     break
            # break
        print('------------------1file---------------------')
    finalDataframe.to_csv('imputation_ChandDataset2_Results.tsv', sep='\t')




