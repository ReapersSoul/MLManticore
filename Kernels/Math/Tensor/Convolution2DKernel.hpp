#pragma once
#include <string>

static std::string Convoution2D_Kernel_Code = R"(


void kernel Convolution2D(global float* X, global float* K, global float* Y, int XSizeX, int XSizeY, int KSizeX, int KSizeY, int YSizeX, int YSizeY, int StrideX, int StrideY, int PaddingX, int PaddingY)
{
int gid_x = get_global_id(0);
int gid_y = get_global_id(1);
int threadCount_x = get_global_size(0);
int threadCount_y = get_global_size(1);

// Starting index of the output for this thread
int StartingOutput_x = gid_x * (YSizeX / threadCount_x);
int StartingOutput_y = gid_y * (YSizeY / threadCount_y);

// Number of outputs to calculate for this thread
int OutputsToCalculate_x = (YSizeX / threadCount_x);
int OutputsToCalculate_y = (YSizeY / threadCount_y);

if (gid_x == threadCount_x - 1) {
OutputsToCalculate_x += YSizeX % threadCount_x;
}
if (gid_y == threadCount_y - 1) {
OutputsToCalculate_y += YSizeY % threadCount_y;
}

for (int x = StartingOutput_x; x < StartingOutput_x + OutputsToCalculate_x; x++) {
for (int y = StartingOutput_y; y < StartingOutput_y + OutputsToCalculate_y; y++) {
int index = y * YSizeX + x;
Y[index] = 0;

for (int k = 0; k < KSizeX; k++) {
for (int l = 0; l < KSizeY; l++) {
int kernelIndex = l * KSizeX + k;

// Calculate the input index based on the output and kernel positions
int InputIndexX = x * StrideX - PaddingX + k;
int InputIndexY = y * StrideY - PaddingY + l;

// Check if the input index is within the bounds of the input data
if (InputIndexX >= 0 && InputIndexX < XSizeX && InputIndexY >= 0 && InputIndexY < XSizeY) {
int InputIndex = InputIndexY * XSizeX + InputIndexX;
Y[index] += X[InputIndex] * K[kernelIndex];
}
}
}
}
}
}
)";

static std::string Convoution2DBackProp_Kernel_Code = R"(


void kernel Convolution2DBackProp(global const float* X, global float* K, global float* ZPrime,global float* dY_dX, const int XSizeX, const int XSizeY, const int KSizeX, const int KSizeY, const int YSizeX, const int YSizeY, const int StrideX, const int StrideY, const int PaddingX, const int PaddingY, const int GradientClipping, const float GradientMax, const float GradientMin)
{
int gid_x = get_global_id(0);
int gid_y = get_global_id(1);
int threadCount_x = get_global_size(0);
int threadCount_y = get_global_size(1);

// Starting index of the output for this thread
int StartingOutput_x = gid_x * (YSizeX / threadCount_x);
int StartingOutput_y = gid_y * (YSizeY / threadCount_y);

// Number of outputs to calculate for this thread
int OutputsToCalculate_x = (YSizeX / threadCount_x);
int OutputsToCalculate_y = (YSizeY / threadCount_y);

if (gid_x == threadCount_x - 1) {
OutputsToCalculate_x += YSizeX % threadCount_x;
}
if (gid_y == threadCount_y - 1) {
OutputsToCalculate_y += YSizeY % threadCount_y;
}

for (int i = StartingOutput_x; i < StartingOutput_x + OutputsToCalculate_x; i++) {
for (int y = StartingOutput_y; y < StartingOutput_y + OutputsToCalculate_y; y++) {
for (int k = 0; k < KSizeX; k++) {
for (int l = 0; l < KSizeY; l++) {
int kernelIndex = l * KSizeX + k;

// Calculate the input index based on the output and kernel positions
int InputIndexX = i * StrideX - PaddingX + k;
int InputIndexY = y * StrideY - PaddingY + l;

// Check if the input index is within the bounds of the input data
if (InputIndexX >= 0 && InputIndexX < XSizeX && InputIndexY >= 0 && InputIndexY < XSizeY) {
int InputIndex = InputIndexY * XSizeX + InputIndexX;
dY_dX[InputIndex] += ZPrime[y * YSizeX + i] * K[kernelIndex];
K[kernelIndex] += ZPrime[y * YSizeX + i] * X[InputIndex];
if(GradientClipping){
if(fabs(K[kernelIndex]) > GradientMax){
K[kernelIndex] = GradientMax;
}
if(fabs(K[kernelIndex]) < GradientMin){
K[kernelIndex] = GradientMin;
}
if(fabs(dY_dX[InputIndex]) > GradientMax){
dY_dX[InputIndex] = GradientMax;
}
if(fabs(dY_dX[InputIndex]) < GradientMin){
dY_dX[InputIndex] = GradientMin;
}
}
}
}
}
}
}
}
)";