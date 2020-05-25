import numpy as np
from scipy import ndimage, signal
import skimage.draw, skimage.transform
import math
from matplotlib import pyplot as plt


class Segmentation:
    """
    An implementation of "A new robust method for blood vessel segmentation in
    retinal fundus images based on weighted line detector and hidden Markov
    model" by Chao Zhou, Xiaogang Zhang, Hua Chen
    https://doi.org/10.1016/j.cmpb.2019.105231
    
    Attributes
    ----------
    w : int
        size of line detection window
    angle : int 
        line detection angle step (degrees)
    k1 : float
        weight of line detection in combined image
    k2 : float
        weight of inverted green channel image in combined image
    t : float
        threshold
    block : int
        block size for searching seeds (tracing start points)
    L : int
        tracing segment lenght
    eps : float
        tracing segment lenght range
    t1 : int
        tracing direction range (degrees)
    e1 : int
        tracing end window size
    e2 : int
        tracing end window size
    a : int
        denoising window size
    p1 : float
        denoising parameter
    p2 : float
        denoising parameter
    img
        original image    
    igc
        inverted green channel image    
    basicimg
        image resulting from basic line detector
    weightedimg
        image resulting from weighted line detector
    combinedimg
        weightedimg combined with igc
    binthresholdimg
        binary thresholded combinedimg image
    lowthresholdimg
        thresholded combinedimg image (values less than t are set to 0)
    tracingimg
        image resulting from tracing vessel certerlines
    unionimg
        binthresholdimg combined with tracingimg
    
    Methods
    -------
    basicLineDetector()
        Returns an image of detected lines. Shows the major vessel structure.
        The resulting image is stored in basicimg
    weightedLineDetector()
        Returns an image of detected lines. Shows the major vessel structure.
        The detector assigns a different weight for each pixel of dectected
        lines based on distance. The resulting image is stored in weightedimg
    combined(img1=None, img2=None)
        By default combines weightedimg and igc (weighted sum). The resulting
        image is stored in combinedimg
    binThreshold(img)
        Return a boolean image were each pixel from combinedimg (by default) is
        returned as True if greater than t and False otherwise. The resulting
        image is stored in binthresholdimg
    lowThreshold()
        Return a boolean image were each pixel from combinedimg (by default) is
        returned as 0 if less than t. The resulting image is stored in
        lowthresholdimg
    tracing()
        Returns a binary image of detected lines. Shows the major vessel
        centerlines. The resulting image is stored in tracingimg
    denoising(img)
        Sets to 0 pixels detected as noise due to the optic disk or dark
        regions. The resulting image is also returned
    union(img1=None, img2=None)
        Given two boolean images (binthresholdimg and tracingimg by default)
        returns their union (OR). The resulting image is stored in unionimg
    """
    
    def __init__(self, imgPath:str, maskPath:str, w=15, angle=15, k1=.67, k2=.33, t=.31, block=11, L=7, eps=.5, t1=90, e1=7, e2=5, a=9, p1=.3, p2=.14):
        self.w = w                  # size of line detection window
        self.angle = angle          # line detection angle step
        self.k1 = k1                # weight of line detection in combined image
        self.k2 = k2                # weight of inverted green channel image in combined image
        self.t = t                  # threshold
        self.block = block          # block size for searching seeds (tracing start points)
        self.L = L                  # tracing segment lenght
        self.eps = eps              # tracing segment lenght range
        self.t1 = t1                # tracing direction range
        self.e1 = e1                # tracing end window size
        self.e2 = e2                # tracing end window size
        self.a = a                  # denoising window size
        self.p1 = p1                # denoising parameter
        self.p2 = p2                # denoising parameter
        
        self.basicimg = None
        self.weightedimg = None
        self.combinedimg = None
        self.binthresholdimg = None
        self.lowthresholdimg = None
        self.tracingimg = None
        self.unionimg = None
        
        # get image
        self.img = plt.imread(imgPath)
        
        # generate inverted green channel image
        self.igc=1-self.img[:,:,1]
        self.Y,self.X = self.igc.shape
        
        # create mask
        self.mask = plt.imread(maskPath)
        self.mask = ~np.array(self.mask,dtype=bool)
        
        # apply mask
        # mean = self.igc.mean()
        # for x in range(self.X):
        #     for y in range(self.Y):
        #         if self.mask[y,x]:
        #             self.igc[y,x]=mean
                    
        self.igc = self.__normalizeData(self.igc)

        
    def __lineMask(self,a):
        
        def line(m, x, o, q):
            return m*(x-o)+q
        
        matrix = np.zeros((self.w, self.w), dtype=bool)
        
        if abs(a) <= 45:
            m = math.tan(math.radians(a))
            for x in range(self.w):
                y = int(round(line(m,x,int((self.w-1)/2),int((self.w-1)/2))))
                if y in range(self.w):
                    matrix[self.w-1-y,x] = True
        else:
            a = (90-abs(a))*np.sign(a)
            m = math.tan(math.radians(a))
            for x in range(self.w):
                y = int(round(line(m,x,int((self.w-1)/2),int((self.w-1)/2))))
                if y in range(self.w):
                    matrix[self.w-1-y,x] = True
            matrix = np.rot90(np.flip(matrix,0))
        
        return matrix
    
    def __normalizeData(self,data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    def basicLineDetector(self):
        
        limit = int(np.ceil(self.w/2))        
        lineMasks = np.array([self.__lineMask(a) for a in range(-90,90,self.angle)])
        
        def R_b(y,x):
            # creating window
            yIndex, xIndex = np.ogrid[y-int((self.w-1)/2):y+int((self.w+1)/2),x-int((self.w-1)/2):x+int((self.w+1)/2)]
            
            # calculating all lines averages
            lineAvg = np.array([self.igc[yIndex,xIndex][m].mean() for m in lineMasks])
                        
            Iw_avg = self.igc[yIndex,xIndex].mean()  # window average
            Iw_max = lineAvg.max()  # max line average
                
            return Iw_max-Iw_avg
        
        # generate filtered image
        filtered = np.zeros(self.igc.shape)  # filtered image
        for x in range(limit,self.X-limit):
            for y in range(limit,self.Y-limit):
                if not self.mask[y,x]:
                    value = R_b(y,x)
                    if value > 0:
                        filtered[y,x]=value
                        
        # cut off borders
        self.mask = ndimage.binary_dilation(self.mask, iterations=limit)
        for x in range(self.X):
              for y in range(self.Y):
                  if self.mask[y,x]:
                      filtered[y,x]=0
        
        filtered = self.__normalizeData(filtered)
        self.basicimg = filtered
    
        return self.basicimg
    
    def weightedLineDetector(self):
        
        limit = int(np.ceil(self.w/2))        
        lineMasks = np.array([self.__lineMask(a) for a in range(-90,90,self.angle)])
        
        def R_w(y,x):
            # creating window
            yIndex, xIndex = np.ogrid[y-int((self.w-1)/2):y+int((self.w+1)/2),x-int((self.w-1)/2):x+int((self.w+1)/2)]
            
            # calculating weights
            weights = np.concatenate((np.arange(1,limit),np.arange(limit,0,-1)))
            
            # calculating all lines averages
            lineAvg = np.array([np.average(self.igc[yIndex,xIndex][m],weights=weights) for m in lineMasks])
                        
            Iw_avg = self.igc[yIndex,xIndex].mean()  # window average
            Iw_max = lineAvg.max()  # max line average
                
            return Iw_max-Iw_avg
        
        # generate filtered image
        filtered = np.zeros(self.igc.shape)  # filtered image
        for x in range(limit,self.X-limit):
            for y in range(limit,self.Y-limit):
                if not self.mask[y,x]:
                    value = R_w(y,x)
                    if value > 0:
                        filtered[y,x]=value

        # cut off borders
        self.mask = ndimage.binary_dilation(self.mask, iterations=1)
        for x in range(self.X):
              for y in range(self.Y):
                  if self.mask[y,x]:
                      filtered[y,x]=0
                        
        filtered = self.__normalizeData(filtered)
                      
        self.weightedimg = filtered
    
        return self.weightedimg
    
    def combined(self,img1=None, img2=None):
        
        if img1 is None:
            img1 = self.weightedimg
            
        if img1 is None:
            img1 = self.weightedLineDetector()
        
        if img2 is None:
            img2 = self.igc
        
        combined = self.__normalizeData(self.k1*img1 + self.k2*img2)
        #combined = self.k1*img1 + self.k2*img2
        
        # cut off borders
        for x in range(self.X):
              for y in range(self.Y):
                  if self.mask[y,x]:
                      combined[y,x]=0
        
        self.combinedimg = combined
        return self.combinedimg
    
    def binThreshold(self, img=None):
        
        if img is None:
            img = self.combinedimg
        
        if img is None:
            img = self.combined()
            
        self.binthresholdimg = img>self.t
        
        return self.binthresholdimg
    
    def lowThreshold(self, img=None):
        
        if img is None:
            img = self.combinedimg
        
        if img is None:
            img = self.combined()
            
        self.lowthresholdimg = self.__normalizeData(np.clip(img,self.t,None))
        
        return self.lowthresholdimg
    
    def __seeds(self):
        
        self.seedx = []
        self.seedy = []
        
        if self.lowthresholdimg is None:
            self.lowThreshold()
        img = self.lowthresholdimg
        # img = self.igc
        low = img.min()
        
        for x in np.arange(0,self.X-self.block,self.block):
            for y in np.arange(0,self.Y-self.block,self.block):
                i = img[y:y+self.block,x:x+self.block].argmax()
                sy, sx = np.unravel_index(i,(self.block,self.block))
                sx+=x
                sy+=y
                
                if not self.mask[sy,sx] and img[sy,sx] != low:
                    self.seedx.append(sx)
                    self.seedy.append(sy)
        
        self.seedx = np.array(self.seedx) 
        self.seedy = np.array(self.seedy) 
                
        return (self.seedx,self.seedy)
        
    def tracing(self):
        
        if self.lowthresholdimg is None:
            self.lowThreshold()
        img = self.lowthresholdimg
        # img = self.igc
        
        def circleMask():
            size = int(self.L*2+self.eps*2)
            center = int(self.L+self.eps-.5)
            mask = np.zeros((size,size))   
            mask[skimage.draw.circle(center, center, self.L+self.eps)]=1
            mask[skimage.draw.circle(center, center, self.L-self.eps)]=0
            
            return np.array(mask,dtype=bool)
        
        def angleMask(a):
            ar = math.radians(a)
            t1r = math.radians(self.t1)
            
            size = int(self.L*2+self.eps*2)
            mask = np.zeros((size,size))
            center = int(self.L+self.eps-.5)
            x=np.array(center)
            y=np.array(center)
            
            for r in np.linspace(ar-t1r,ar+t1r,int(self.t1/10)):
                x = np.append(x,np.cos(r)*(self.L+self.eps)+center)
                y = np.append(y,-np.sin(r)*(self.L+self.eps)+center)
                        
            mask[skimage.draw.polygon(y, x)]=1
            
            return np.array(mask,dtype=bool)
        
        def endMask(a):
            mask = np.ones((self.e2,self.e1))
            mask = skimage.transform.rotate(mask,a,resize=True)
            return np.array(mask,dtype=bool)
        
        def startTracing():
            size = int(self.L*2+self.eps*2)
            hsize = int(size/2)
            sx,sy = self.__seeds()
            tracea = []
            tracex = []
            tracey = []
            
            for x,y in zip(sx,sy):
                temp = img[y-hsize:y+hsize+1,x-hsize:x+hsize+1] 
                mask = circleMask()
                maxima = signal.find_peaks(temp[mask],distance=2)[0]
                
                x_maxima = np.mgrid[0:size,0:size][1][mask][maxima]
                y_maxima = np.mgrid[0:size,0:size][0][mask][maxima]
                                
                for xm,ym in zip(x_maxima,y_maxima):
                    angle = 360-int(np.degrees(np.arctan2(ym-hsize,xm-hsize)))
                    tracex.append(x)
                    tracey.append(y)
                    tracea.append(angle)
                    
            self.startx=np.array(tracex)
            self.starty=np.array(tracey)
            self.starta=np.array(tracea)
            
            
        def endTracing(x,y,a) -> bool:
            mask = endMask(a)
            size = mask.shape
            hsizestart = int(np.floor(size[0]/2)),int(np.floor(size[1]/2))
            hsizestop = int(np.ceil(size[0]/2)),int(np.ceil(size[1]/2))
            
            pixel = img[y,x]
            surrounding = img[y-hsizestart[0]:y+hsizestop[0],x-hsizestart[1]:x+hsizestop[1]].mean()
            
            return pixel <= surrounding# or self.mask[y,x]
        
        def psi(x,x1,y,y1,img=img):
            alpha = 0;
            for k in range(1,self.L+1):
                alpha += img[int(np.rint((k*y+(self.L-k)*y1)/self.L)),int(np.rint((k*x+(self.L-k)*x1)/self.L))]
            return alpha
        
        def stepTracing(x,y,a):
            mask = circleMask() & angleMask(a)
            size = int(self.L*2+self.eps*2)
            hsize = int(size/2)
            temp = img[y-hsize:y+hsize+1,x-hsize:x+hsize+1]
            calc = np.zeros(temp.shape)
                       
            for xi in np.arange(size):
                for yi in np.arange(size):
                    if mask[yi,xi]:
                        calc[yi,xi] = psi(xi,hsize,yi,hsize,temp)  
                    else:
                        calc[yi,xi] = np.NINF
            
            my, mx = np.unravel_index(calc.argmax(),mask.shape)
            ma = (360-int(np.degrees(np.arctan2(my-hsize,mx-hsize))))%360
            mx += x-hsize
            my += y-hsize
            
            return mx,my,ma
                        
        startTracing()
        self.tracingimg = np.zeros(img.shape,dtype=bool)
        
        for i in np.arange(self.startx.size):
            x,y,a = stepTracing(self.startx[i], self.starty[i], self.starta[i])
            while not endTracing(x,y,a):                
                xo,yo = x,y
                x,y,a = stepTracing(x,y,a)
                rr, cc = skimage.draw.line(yo, xo, y, x)
                self.tracingimg[rr,cc] = True
        
        return self.tracingimg
                
    def denoising(self,img):
        size = int(self.a*2)+1
        hsize = int(self.a)
        # denoise = np.copy(img)
        denoise = img
           
        def lineMask(a):
            ad = np.deg2rad(a)
            mask = np.zeros((size,size),dtype=bool)
            x2 = int(np.rint(hsize+hsize*np.cos(ad)))
            y2 = int(np.rint(hsize-hsize*np.sin(ad)))
            x1 = int(np.rint(hsize-hsize*np.cos(ad)))
            y1 = int(np.rint(hsize+hsize*np.sin(ad)))
            rr, cc = skimage.draw.line(y1, x1, y2, x2)
            mask[rr,cc] = 1            
            return mask
        
        def sideMask(a):
            ad0 = np.deg2rad(a)
            ad1 = np.deg2rad(a+180)
            mask = np.zeros((size,size),dtype=bool)
            
            x = []
            y = []
            
            for r in np.linspace(ad0,ad1,10):
                x.append(int(np.rint(hsize+hsize*np.cos(r))))
                y.append(int(np.rint(hsize-hsize*np.sin(r))))
            
            #mask[skimage.draw.polygon_perimeter(y, x)]=True
            mask[skimage.draw.polygon(y, x)]=True
            
            x2 = int(np.rint(hsize+hsize*np.cos(ad1)))
            y2 = int(np.rint(hsize-hsize*np.sin(ad1)))
            x1 = int(np.rint(hsize+hsize*np.cos(ad0)))
            y1 = int(np.rint(hsize-hsize*np.sin(ad0)))
            rr, cc = skimage.draw.line(y1, x1, y2, x2)
            mask[rr,cc] = False
            
            return mask
        
        lineMasks = []
        sideMasks0 = []
        sideMasks1 = []
        
        angles=np.arange(-90,90,15)
        
        for a in angles:
            lineMasks.append(lineMask(a))
            sideMasks0.append(sideMask(a))
            sideMasks1.append(sideMask(a+180))
        
        
        def isNoise(x,y) -> bool:
            
            temp = self.igc[y-hsize:y+hsize+1,x-hsize:x+hsize+1]
               
            #angles=np.arange(-90,90,15)
            means=np.zeros(angles.shape)
            
            for i in np.arange(angles.size):
                means[i] = temp[lineMasks[i]].mean()
            
            i = means.argmax()
            #a = angles[i]
            I = temp[hsize,hsize]
            
            m1=temp[sideMasks0[i]].min()
            m2=temp[sideMasks1[i]].min()
            
            M = np.abs(m1-m2)>self.p1
            N = not ((I-m1)>self.p2 and (I-m2)>self.p2)
            
            if not M:
                return False
            elif N:
                return True
                    
        for x in np.arange(size,self.X-size):
            for y in np.arange(size,self.Y-size):
                if not self.mask[y,x]:
                    if isNoise(x, y):
                        denoise[y,x] = 0
        
        return denoise

    def union(self,img1=None, img2=None):
        if img1 is None:
            img1 = self.binthresholdimg
            
        if img1 is None:
            img1 = self.binThreshold()
        
        if img2 is None:
            img2 = self.tracingimg
            
        if img2 is None:
            img2 = self.tracing()
            
        self.unionimg = img1 | img2
            

if __name__ == "__main__":
    sgmt = Segmentation('01_test.png','01_test_mask.gif')
    
    print("Original image:")
    plt.imshow(sgmt.img, cmap="Greys_r")
    plt.show()
    
    print("\nInverted green channel:")
    plt.imshow(sgmt.igc, cmap="Greys_r")
    plt.show()
    
    print("\nMajor structure of vessels (line detector):")
    sgmt.binThreshold()
    plt.imshow(sgmt.binthresholdimg, cmap="Greys_r")
    plt.show()
    
    print("\nMajor structure of vessels after postprocessing:")
    sgmt.denoising(sgmt.binthresholdimg)
    plt.imshow(sgmt.binthresholdimg, cmap="Greys_r")
    plt.show()
    
    print("\nVessel centerlines (vessel tracing using a HMM):")
    sgmt.tracing()
    plt.imshow(sgmt.tracingimg, cmap="Greys_r")
    plt.show()
    
    print("\nVessel centerlines after postprocessing:")
    sgmt.denoising(sgmt.tracingimg)
    plt.imshow(sgmt.tracingimg, cmap="Greys_r")
    plt.show()
    
    print("\nFinal image (union):")
    sgmt.union()
    plt.imshow(sgmt.unionimg, cmap="Greys_r")
    plt.show()
    
