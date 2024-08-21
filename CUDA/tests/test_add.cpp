// tests/test_add.cpp

#include <iostream>
#include <cassert>
#include "../include/add.h"
void test_add() {
    int a[4] = {1, 2, 3, 4};
    int b[4] = {5, 6, 7, 8};
    int c[4] = {0, 0, 0, 0};
    int expected[4] = {6, 8, 10, 12};

    add_arrays(a, b, c, 4);

    for (int i = 0; i < 4; i++) {
        assert(c[i] == expected[i]);
    }

    std::cout << "test_add passed!" << std::endl;
}

int main() {
    test_add();
    return 0;
}
