update:
The bin files has been converted to little-endian format, and the outliers has been reset to 0.
Each bin file contains 100x500x500 FP32 values.


Original data set:
http://vis.computer.org/vis2004contest/data.html
IEEE Visualization 2004 Contest
Data Set

The theme for the inaugural IEEE Visualization contest was visualization fusion. The goal was to provide a data set which will allow researchers to showcase their visualization techniques from a wide range of disciplines. Since the type of information being shown is so diverse, we hoped to encourage collaboration of researchers from all domains of scientific and information visualization.

The data set for this contest is a simulation of a hurricane from the National Center for Atmospheric Research in the United States. The data consists of several time-varying scalar and vector variables over large dynamic ranges. The sheer size of the data also presents a challenge for interactive exploration.
Data Format

A summary of the data can be found below:

    Dimensions: 500 x 500 x 100
    Physical Scale: 2139km (east-west) x 2004km (north-south) x 19.8km (vertical) Note: The east-west distance of 2139km is measured at the southern latitude; the distance is only 1741km at the northern latitude due to the curvature of the earth.
    Physical Location: Longitude (x): 83W to 62W; Latitude (y): 23.7N to 41.7N; Height (z): 0.035km to 19.835km
    Format: Brick-of-Floats
    Number of Variables: 13
    Number of Time Steps: 48 (1 simulated hour between time steps)
    Size per Time Step: ~95 MB uncompressed

Format

The data is in "Brick-of-Floats" format. It consists of a volume of data values at each position in space. The three-dimensional array of data consists of planes of x-y values in ascending z order; in the data, the x values vary fastest. Assuming the data was stored as a one-dimensional array, the index into that array for the point x, y, z would be:
index=x+dim_x×(y+dim_y×z)

where dim_x is the number of x values (500) and dim_y is the number of y values (500).
Variables

The following table described the variables in the data set. Each variable is contained in a separate data file, one for each time step. All data values are floating-point values.
Variable 	Description 	Range
QCLOUD 	Cloud moisture mixing ratio (kg water/kg dry air) 	0.00000/0.00332
QGRAUP 	Graupel mixing ratio (solid ice precipitation: hail, sleet, snow pellets) 	0.00000/0.01638
QICE 	Cloud ice mixing ratio 	0.00000/0.00099
QSNOW 	Snow mixing ratio 	0.00000/0.00135
QVAPOR 	Water vapor mixing ratio 	0.00000/0.02368
CLOUD 	Total cloud moisture mixing ratio (QCLOUD+QICE) 	0.00000/0.00332
PRECIP 	Total precipitation mixing ratio (QGRAUP+QRAIN+QSNOW) 	0.00000/0.01672
P 	Pressure (weight of atmosphere above a grid point) 	-5471.85791/3225.42578
TC 	Temperature (Celsius) 	-83.00402/31.51576
U 	X wind speed (positive means winds from west to east) 	-79.47297/85.17703
V 	Y wind speed (positive means winds from south to north) 	-76.03391/82.95293
W 	Z wind speed (positive means upward wind) 	-9.06026/28.61434

Note that all data is stored in big-endian format and compressed via GNU zip (gzip) for storage. Each file is named VARfNN.bin.gz where VAR is the variable name and NN is the time step.
Physical Position

Since the data in question simulates an actual event (a hurricane), each computational data point (voxel) corresponds to an actual physical point. The surface topology (for the actual ground) is in a special 500x500x1 data file (HGTdata.bin.gz). In the other files, where there is ground, no data is recorded. The special value for this "no data" value is 1.0000000e+35.
Acquiring the Data

The URL for the data is: http://www.vets.ucar.edu/vg/isabeldata/. A README file with more information and a Python script to simplify downloading the data are also available.

If you use the data set, please provide the following attribution:

    The authors will like to thank Bill Kuo, Wei Wang, Cindy Bruyere, Tim Scheitlin, and Don Middleton of the U.S. National Center for Atmospheric Research (NCAR) and the U.S. National Science Foundation (NSF) for providing the Weather Research and Forecasting (WRF) Model simulation data of Hurricane Isabel. 

A shorter attribution is:

    Hurricane Isabel data produced by the Weather Research and Forecast (WRF) model, courtesy of NCAR and the U.S. National Science Foundation (NSF). 

For More Information

For for information about hurricanes or the method used to generate the data, consult the following links.

    The hurricane modeled is Hurricane Isabel from September of 2003. It was a very strong hurricane in the west Atlantic region.
    Tropical Cyclones from the University of Wisconsin
    Tropical Cyclone Archive from the National Weather Service
    The Weather Research and Forecasting (WRF) Model was used to generate the data

Acknowledgments

The data set has kindly been provided by Wei Wang, Cindy Bruyere, Bill Kuo, and others at NCAR. Tim Scheitlin at NCAR converted the data into the Brick-of-Float format described above.
