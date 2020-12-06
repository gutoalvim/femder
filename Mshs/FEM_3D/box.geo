// Gmsh project created on Fri Dec 04 19:40:15 2020
SetFactory("OpenCASCADE");
//+
Box(1) = {-0.5, -0.5, 0, 1, 1, 1};
//+
Physical Volume(1) = {1};
//+
Physical Surface(2) = {3, 1, 4, 6, 2, 5};
