#pragma once
#include <string>

static std::string CalcZs_Kernel_Code = R"(


void kernel CalcZs(global float* X, int XSize, global float* W, int WSizeX, int WSizeY, global float* B, int BSize, global float* Z, int ZSize)
{
int gid = get_global_id(0);
int threads = get_global_size(0);
//starting index for this thread
int startingIndex = gid * (ZSize/threads);
//number of z's to calculate
int Zs_to_calculate = ZSize / threads;
if(gid==threads-1)
{
Zs_to_calculate += ZSize % threads;
}

for(int i=startingIndex; i<startingIndex+Zs_to_calculate; i++)
{
Z[i] = 0;
for(int j=0; j<XSize-1; j++)
{
Z[i] += X[j] * W[j*WSizeX + i];
}
Z[i] += B[i];
}
}
)";

static std::string Backward_Kernel_Code = R"(


void kernel Backward(global float* X, int XSize, global float* W, int WSizeX, int WSizeY, global float* B, int BSize, global float* ZPrime, int ZPrimeSize, global float* XPrime, int XPrimeSize, int GradientClipping, float GradientMax, float GradientMin)
{
int gid=get_global_id(0);
int threads = get_global_size(0);
//starting index for this thread
int startingIndex = gid * (ZPrimeSize/threads);
//number of x's to calculate
int Zs_to_calculate = ZPrimeSize / threads;
if(gid==threads-1)
{
Zs_to_calculate += ZPrimeSize % threads;
}

for(int i=startingIndex; i<startingIndex+Zs_to_calculate; i++)
{
for(int j=0; j<XSize-1; j++)
{
XPrime[j] += W[j*WSizeX + i] * ZPrime[i];
W[j*WSizeX + i] += X[j] * ZPrime[i];
if(GradientClipping==1)
{
if(fabs(XPrime[j]) > GradientMax)
{
XPrime[j] = GradientMax;
}
else if(fabs(XPrime[j]) < GradientMin)
{
XPrime[j] = GradientMin;
}
if(fabs(W[j*WSizeX + i]) > GradientMax)
{
W[j*WSizeX + i] = GradientMax;
}
else if(fabs(W[j*WSizeX + i]) < GradientMin)
{
W[j*WSizeX + i] = GradientMin;
}
}
}
B[i] += ZPrime[i];
if(GradientClipping==1)
{
if(fabs(B[i]) > GradientMax)
{
B[i] = GradientMax;
}
else if(fabs(B[i]) < GradientMin)
{
B[i] = GradientMin;
}
}
}
}
)";