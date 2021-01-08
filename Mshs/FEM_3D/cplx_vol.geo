Merge "cplx_room.brep";
Physical Volume(1) = {1};
//+
Physical Surface(2) = {4};
//+
Physical Surface(3) = {2, 7, 1, 6, 10, 3, 9, 8, 5};

//+
SetFactory("OpenCASCADE");
Box(2) = {-1, 0.2, 2, 2, 2, 0.5};
//+
Physical Volume(4) = {2};
//+
BooleanDifference{ Volume{1}; Surface{3}; Delete; }{ Surface{15}; Surface{16}; Surface{12}; Surface{13}; Curve{31}; Volume{2}; Surface{11}; Surface{14}; }
//+
BooleanDifference{ Volume{1}; }{ Volume{2}; }
