#include <iostream>

void vecAdd(float* A_h, float* B_h, float* C_h, int n)
{
    for (int i = 0; i < n; i++)
    {
        C_h[i] = A_h[i] + B_h[i];
    }
}

int main()
{
    std::cout << "Naive Vector Addition" << std::endl;

    int n = 1 << 5;

    float* A_h = new float[n] ();
    float* B_h = new float[n];
    float* C_h = new float[n];

    for (int i = 0; i < n; i++)
    {
        A_h[i] = i*1.0f;
        B_h[i] = i*2.0f;
    }

    vecAdd(A_h, B_h, C_h, n);

    std::cout << "A: ";
    for (int i = 0; i < n; i++)
    {
        std::cout << A_h[i] << " ";
    }

    std::cout << std::endl << "B: ";
    for (int i = 0; i < n; i++)
    {
        std::cout << B_h[i] << " ";
    }

    std::cout << std::endl << "C: ";
    for (int i = 0; i < n; i++)
    {
        std::cout << C_h[i] << " ";
    }

    std::cout << std::endl;

}