# imgUtils.py
# 16.02.2023
# m.biester

import sys
import numpy as np
from skimage.metrics import structural_similarity as stsim


class imgSimCheck:
    
    def __init__(self, imgNdarray, nrow=8, ncol=8, sim_threshold = 0.9):
        """_summary_

        Args:
            imgNdarray (numpy matrix): eg. data as provided by OpenCv ; cv2.imread
            nrow (int, optional): imgNdarray is partitioned subimages. There are ncol subareas per number
                                    of rows. Defaults to 8.
            ncol (int, optional): number of rows. Defaults to 8.
            
                                in total there ncol*nrow subimages
            sim_threshold (float) : threshold in [0, 1);
        """
        self.img1 = imgNdarray
        self.nrow = nrow
        self.ncol = ncol
        self.sim_threshold = sim_threshold
        self.height, self.width, self.chi = imgNdarray.shape
        self.is_color_img = True if self.chi == 1 else False
        self.n_col_step = self.width // self.ncol
        self.n_row_step = self.height // self.nrow
        self.scores = np.zeros(shape=(self.nrow, self.ncol))
        # list of coordinates where similarity < threshold
        self.lstLtThreshold = []
        
    def checkSimilarity(self, newImg, new_sim_threshold=None) -> bool:
        if new_sim_threshold is not None:
            if (new_sim_threshold >=0) and (new_sim_threshold <=1):
                # use updated threshold
                self.sim_threshold = new_sim_threshold
            else:
                print(f"incompatible value new_sim_threshold: {new_sim_threshold}; expected value in [0, 1]")
       
        # check image size; images must have identical size to apply similarity check ...
        if self.img1.shape != newImg.shape:
            sys.exit(f"images have different shape: {self.img1.shape} != {newImg.shape}")
            
        # list of rectangle coordinates where similarity < threshold
        self.scores = np.zeros(shape=(self.nrow, self.ncol))
        self.lstLtThreshold = []
        images_are_different = False
        row_start = 0  
                        
        for m in range(self.nrow):
            row_start = m * self.n_row_step
            row_end = row_start + self.n_row_step - 1
            
            for k in range(self.ncol):
                col_start = k * self.n_col_step
                col_end = col_start + self.n_col_step - 1
                
                score_channels = [stsim(self.img1 [row_start:row_end,col_start:col_end, ch_i], newImg[row_start:row_end,col_start:col_end, ch_i]) for ch_i in range(self.chi)]
                self.scores[m, k] = min(score_channels)
                if self.scores[m, k] < self.sim_threshold:
                    images_are_different = True
                    # list of list (coordinates : upper-left corner, coordinates : lower right corner)
                    self.lstLtThreshold.append([(col_start, row_start), (col_end, row_end)])
        
        # copy new image to old image 
        self.img1 = np.copy(newImg)
        return images_are_different

class imgDiffCheck:
    
    def __init__(self, imgNdarray, nrow=8, ncol=8, sim_threshold = 0.9):
        """_summary_

        Args:
            imgNdarray (numpy matrix): eg. data as provided by OpenCv ; cv2.imread
            nrow (int, optional): imgNdarray is partitioned subimages. There are ncol subareas per number
                                    of rows. Defaults to 8.
            ncol (int, optional): number of rows. Defaults to 8.
            
                                in total there ncol*nrow subimages
            sim_threshold (float) : threshold in [0, 1);
        """
        self.img1 = imgNdarray
        self.nrow = nrow
        self.ncol = ncol
        self.sim_threshold = sim_threshold
        self.height, self.width, self.chi = imgNdarray.shape
        self.is_color_img = True if self.chi == 1 else False
        self.n_col_step = self.width // self.ncol
        self.n_row_step = self.height // self.nrow
        self.scores = np.zeros(shape=(self.nrow, self.ncol))
        # list of coordinates where similarity < threshold
        self.lstLtThreshold = []
        
    def checkDiff(self, newImg, new_sim_threshold=None) -> bool:
        if new_sim_threshold is not None:
            if (new_sim_threshold >=0) and (new_sim_threshold <=1):
                # use updated threshold
                self.sim_threshold = new_sim_threshold
            else:
                print(f"incompatible value new_sim_threshold: {new_sim_threshold}; expected value in [0, 1]")
       
        # check image size; images must have identical size to compute difference ...
        if self.img1.shape != newImg.shape:
            sys.exit(f"images have different shape: {self.img1.shape} != {newImg.shape}")
            
        # list of rectangle coordinates where similarity < threshold
        self.scores = np.zeros(shape=(self.nrow, self.ncol))
        self.lstLtThreshold = []
        images_are_different = False
        row_start = 0  
                        
        for m in range(self.nrow):
            row_start = m * self.n_row_step
            row_end = row_start + self.n_row_step - 1
            
            for k in range(self.ncol):
                col_start = k * self.n_col_step
                col_end = col_start + self.n_col_step - 1
                
                # compute differences in sub-image
                img_diff = self.img1[row_start:row_end,col_start:col_end,:] - newImg[row_start:row_end,col_start:col_end,:]
                # score : ratio of zero entries to size of sub-images -> [0, 1)]
                score = 1.0 - (np.count_nonzero(img_diff)/img_diff.size)
                self.scores[m, k] = score
                if score < self.sim_threshold:
                    images_are_different = True
                    # list of list (coordinates : upper-left corner, coordinates : lower right corner)
                    self.lstLtThreshold.append([(col_start, row_start), (col_end, row_end)])
        
        # copy new image to old image 
        self.img1 = np.copy(newImg)
        return images_are_different    

# internal test
if __name__ == "__main__":
    
    from argparse import ArgumentParser
    import cv2
    import time
    
    parser = ArgumentParser()
    parser.add_argument("imgFileInp1", help="full path to image file (Inp) Nr. 1")
    parser.add_argument("imgFileInp2", help="full path to image file (Inp) Nr. 2")
    parser.add_argument("imgFileOut_sim", help="full path to image file (Out) processed with structural similarity")
    parser.add_argument("imgFileOut_diff", help="full path to image file (Out) processed with difference")
    
    args = parser.parse_args()
    imgFileInp1 = args.imgFileInp1
    imgFileInp2 = args.imgFileInp2 
    imgFileOut_sim = args.imgFileOut_sim
    imgFileOut_diff = args.imgFileOut_diff
    print(f"imgFileInp1     : {imgFileInp1}") 
    print(f"imgFileInp2     : {imgFileInp2}") 
    print(f"imgFileOut_sim  : {imgFileOut_sim}") 
    print(f"imgFileOut_diff : {imgFileOut_diff}") 
    
    # check with structural similarity algorithm
    imgInp1 = cv2.imread(imgFileInp1)
    imgInp2 = cv2.imread(imgFileInp2)
    
    t1 = time.perf_counter()
    imgChecker = imgSimCheck(imgNdarray=imgInp1, sim_threshold=0.88)
    checkResult = imgChecker.checkSimilarity(imgInp2)
    
    # overlay rectangles where images are significantly different
    # use green frames
    for upper_left, lower_right in imgChecker.lstLtThreshold:
        cv2.rectangle(imgInp2, upper_left, lower_right, (0, 255, 0), 1)

    # create new image and save to file        
    cv2.imwrite(imgFileOut_sim, imgInp2)
    
    t_elapsed = time.perf_counter() - t1
    print(f"time (elapsed) structural similarity algorithm : {t_elapsed:8.3f} seconds")
    
    # check with differences algorithm
    t3 = time.perf_counter()
    imgCheckerDiff = imgDiffCheck(imgNdarray=imgInp1, sim_threshold=0.88)
    checkResultDiff = imgCheckerDiff.checkDiff(imgInp2)
    
    # overlay rectangles where images are significantly different
    # use green frames
    for upper_left, lower_right in imgCheckerDiff.lstLtThreshold:
        cv2.rectangle(imgInp2, upper_left, lower_right, (0, 255, 0), 1)

    # create new image and save to file        
    cv2.imwrite(imgFileOut_diff, imgInp2)
    
    t_elapsed = time.perf_counter() - t3
    print(f"time (elapsed) difference algorithm : {t_elapsed:8.3f} seconds")

