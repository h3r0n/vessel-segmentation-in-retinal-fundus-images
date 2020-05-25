# Vessel segmentation in retinal fundus images

A Python3/NumPy/SciPy implementation of 
> A new robust method for blood vessel segmentation in
> retinal fundus images based on weighted line detector and hidden Markov
> model" by Chao Zhou, Xiaogang Zhang, Hua Chen https://doi.org/10.1016/j.cmpb.2019.105231

Tested on the DRIVE (Digital Retinal Images for Vessel Extraction) dataset

From the Segmentation class docstring:

```
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
```
