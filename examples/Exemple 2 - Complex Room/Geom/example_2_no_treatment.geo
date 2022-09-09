// Gmsh project created on Tue Jul 26 16:11:18 2022
SetFactory("OpenCASCADE");
//+
Box(1) = {-3.67/2, -4.77/2, 0, 3.67, 4.77, 3.4};
//+
Translate {0, 4.77/2, 0} {
  Curve{12}; Curve{2}; Curve{10}; Curve{6}; Curve{3}; Curve{7}; Curve{11}; Curve{1}; Curve{4}; Curve{5}; Curve{8}; Curve{9}; Surface{6}; Surface{4}; Surface{1}; Surface{2}; Surface{3}; Surface{5}; Volume{1}; 
}
//+
Physical Volume(1) = {1};
//+
Physical Surface(2) = {4, 2, 6, 5, 1, 3};
