In this folder you can find as an example how to measure a few phenotypes from a DRIVE image (image 17).

To avoid using LWNET, ARIA, and OD software. The outputs are going to be provided already: 

- The Artery Vein segmentation produced by LWNET ('17_test_seg.png' and '17_test.png'). 

- ARIA output are the X,Y coordinates of each vessel pixel ('17_test_all_center1Coordinates.tsv' and '17_test_all_center2Coordinates.tsv'), the diameter of the vessel at each centerline pixel ('17_test_all_rawDiameters.tsv'), and how these vessel segments are grouped into vessel segments (each row of the previous files is a segment). And '17_test_all_segmenStats.tsv' contain information of each of these vessel segments, e.g. first colums and first row of this file (medianDiameter), has as a value the median of all the diameters of this segment, i.e. es equivalent to compute the median of the first row of '17_test_all_rawDiameters.tsv'. The column 'AVScore' show you the coinfiance of a segment being an artery or a vein. Negatives numbers are arteries and positives numbers are veins. 

- 'od_all.csv' has the results of applying the OD position code to the DRIVE dataset.

All these files can be generated using the code in /... but here we are going to use them as an input to measure some phenotypes of relevance.

