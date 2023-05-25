#include <CL/sycl.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <time.h>
#include <sys/time.h>

using namespace sycl;

static const int N = 4096;
typedef float ele_t;
ele_t mat[N][N];

void LU(ele_t mat[N][N], int n)
{
    ele_t(*new_mat)[N] = (ele_t(*)[N])malloc(N * N * sizeof(ele_t));
    memcpy(new_mat, mat, sizeof(ele_t) * N * N);

    timespec start, end;
    double time_used = 0;
    clock_gettime(CLOCK_REALTIME, &start);

    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
        {
            if (new_mat[i][i] == 0)
                continue;
            ele_t div = new_mat[j][i] / new_mat[i][i];
            for (int k = i; k < n; k++)
                new_mat[j][k] -= new_mat[i][k] * div;
        }

    clock_gettime(CLOCK_REALTIME, &end);
    time_used += end.tv_sec - start.tv_sec;
    time_used += double(end.tv_nsec - start.tv_nsec) / 1000000000;
    std::cout << "串行算法用时: " << time_used << std::endl;

    if (n > 16)
        return;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            std::cout << new_mat[i][j] << ' ';
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void LU_gpu(ele_t mat[N][N], int n)
{
    queue q;

    std::cout << "并行算法使用设备: " << q.get_device().get_info<info::device::name>() << std::endl;

    //# 需要使用共享内存
    ele_t(*new_mat)[N] = (ele_t(*)[N])malloc_shared<ele_t>(N * N, q);

    memcpy(new_mat, mat, sizeof(ele_t) * N * N);
    timespec start, end;
    double time_used = 0;
    clock_gettime(CLOCK_REALTIME, &start);

    for (int i = 0; i < n; i++)
        q.parallel_for(range{ (unsigned long)(n - (i + 1)) }, [=](id<1> idx)
            {
                int j = idx[0] + i + 1;
                ele_t div = new_mat[j][i] / new_mat[i][i];
                for (int k = i; k < n; k++)
                    new_mat[j][k] -= new_mat[i][k] * div;
            }).wait();

            clock_gettime(CLOCK_REALTIME, &end);
            time_used += end.tv_sec - start.tv_sec;
            time_used += double(end.tv_nsec - start.tv_nsec) / 1000000000;
            std::cout << "并行算法用时: " << time_used << std::endl;

            if (n > 16)
                return;

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                    std::cout << new_mat[i][j] << ' ';
                std::cout << std::endl;
            }
            std::cout << std::endl;
}

int main()
{
    std::ifstream data("gauss.dat", std::ios::in | std::ios::binary);
    data.read((char*)mat, N * N * sizeof(ele_t));
    data.close();

    std::cout << "矩阵大小：" << N << " * " << N << std::endl;
    LU(mat, N);
    LU_gpu(mat, N);

    return 0;
}
