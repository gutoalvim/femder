// Gmsh project created on Wed Nov 25 23:30:44 2020
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0.6, 0, 0, 1.0};
//+
Point(3) = {0.8, 0.1, 0, 1.0};
//+
Point(4) = {0.8, 0, 0, 1.0};
//+
Point(5) = {1, 0, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 4};
//+
Line(3) = {4, 3};
//+
Line(4) = {4, 5};
//+
Physical Curve(2) = {3};
//+
Physical Curve(1) = {1, 2, 4};

