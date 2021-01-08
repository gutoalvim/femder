Merge "cplx_room.brep";
Physical Volume(1) = {1};
//+
Physical Surface(2) = {4};
//+
Physical Surface(3) = {2, 7, 1, 6, 10, 3, 9, 8, 5};

//+
Field[1] = Box;
//+
Transfinite Surface {1};
//+
Transfinite Surface {5};
//+
Transfinite Surface {3};
//+
Transfinite Surface {9};
//+
Transfinite Surface {8};
//+
Transfinite Surface {2};
//+
Transfinite Surface {7};
//+
Transfinite Surface {4};
//+
Transfinite Surface {6};
//+
Transfinite Surface {10};
//+
SetFactory("OpenCASCADE");
Box(2) = {-1, 0.5, 2.2, 2, 2, 0.5};
