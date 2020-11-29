// Gmsh project created on Fri Nov 27 13:49:48 2020
SetFactory("OpenCASCADE");
//+
Point(1) = {-0, 0, 0, 1.0};
//+
Point(2) = {48.25, 0, 0, 1.0};
//+
Point(3) = {108.25, 0, 0, 1.0};
//+
Point(4) = {138.25, 0, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Physical Curve(1) = {1, 3};
//+
Physical Curve(2) = {2};
