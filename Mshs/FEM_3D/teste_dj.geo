// Gmsh project created on Tue Mar 16 19:26:54 2021
SetFactory("OpenCASCADE");
//+
Box(1) = {-2, -1.715, 0, 4, 3.43, 2.63};
//+
Physical Volume(1) = {1};
//+
Physical Surface(2) = {5, 6};
//+
Physical Surface(3) = {4, 2, 3, 1};
