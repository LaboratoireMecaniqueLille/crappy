#include <cuda.h>
#define WIDTH %d
#define HEIGHT %d
#define PARAMETERS %d

texture<float, cudaTextureType2D, cudaReadModeElementType> tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> tex_d;

__global__ void gradient(float* gradX, float* gradY)
{
  //Sobel
  uint x = blockIdx.x*blockDim.x+threadIdx.x;
  uint y = blockIdx.y*blockDim.y+threadIdx.y;
  gradX[x+WIDTH*y] = (tex2D(tex,(x+1.5f)/WIDTH,(float)y/HEIGHT)+tex2D(tex,(x+1.5f)/WIDTH,(y+1.f)/HEIGHT)-tex2D(tex,(x-.5f)/WIDTH,(float)y/HEIGHT)-tex2D(tex,(x-.5f)/WIDTH,(y+1.f)/HEIGHT)); // Note: it is not normalized (it will be multiplied later anyways so it would be a waste of time)
  gradY[x+WIDTH*y] = (tex2D(tex,(float)x/WIDTH,(y+1.5f)/HEIGHT)+tex2D(tex,(x+1.f)/WIDTH,(y+1.5f)/HEIGHT)-tex2D(tex,(float)x/WIDTH,(y-.5f)/HEIGHT)-tex2D(tex,(x+1.f)/WIDTH,(y-.5f)/HEIGHT));
}

__global__ void resampleO(float* out, int w, int h)
{
  int idx = threadIdx.x+blockIdx.x*blockDim.x;
  int idy = threadIdx.y+blockIdx.y*blockDim.y;
  if(idx < w && idy < h)
    out[idx+w*idy] = tex2D(tex,(float)idx/w,(float)idy/h);
}

__global__ void resample(float* out, int w, int h)
{
  int idx = threadIdx.x+blockIdx.x*blockDim.x;
  int idy = threadIdx.y+blockIdx.y*blockDim.y;
  if(idx < w && idy < h)
    out[idx+w*idy] = tex2D(tex_d,(float)idx/w,(float)idy/h);
}

__global__ void makeG(float* G, float* gradX, float* gradY, float* fieldX, float* fieldY)
{
  int idx = threadIdx.x+blockIdx.x*blockDim.x;
  int idy = threadIdx.y+blockIdx.y*blockDim.y;
  if(idx < WIDTH && idy < HEIGHT)
  {
    int id = idx+WIDTH*idy;
    G[id] = gradX[id]*fieldX[id]+gradY[id]*fieldY[id];
  }
}

__global__ void makeDiff(float *out, float *param, float *fieldsX, float *fieldsY)
{
  int idx = threadIdx.x+blockIdx.x*blockDim.x;
  int idy = threadIdx.y+blockIdx.y*blockDim.y;
  if(idx < WIDTH && idy < HEIGHT)
  {
    int id = idx+WIDTH*idy;
    float ox = .5f;
    float oy = .5f;
    for(uint i = 0; i < PARAMETERS; i++)
    {
      ox += param[i]*fieldsX[WIDTH*HEIGHT*i+id];
      oy += param[i]*fieldsY[WIDTH*HEIGHT*i+id];
    }
    out[id] = tex2D(tex,(idx+.5f)/WIDTH,(idy+.5f)/HEIGHT)-tex2D(tex_d,(idx+ox)/WIDTH,(idy+oy)/HEIGHT);
  }
}

__global__ void myDot(float *M, float *v)
{
  uint id = threadIdx.x;
  __shared__ float sh_v[PARAMETERS];
  float val = 0;
  sh_v[id] = v[id];
  __syncthreads();
  for(uint i = 0; i < PARAMETERS; i++)
  {
    val += M[id*PARAMETERS+i]*sh_v[i];
  }
  v[id] = val;
}


__global__ void kadd(float* v, float k, float* v2)
{
  v[threadIdx.x] += k*v2[threadIdx.x];
}
