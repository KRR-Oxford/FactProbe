#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <chrono>

#include <sdsl/suffix_array_algorithm.hpp>
#include <sdsl/suffix_arrays.hpp>

using namespace sdsl;
using namespace std;
using namespace std::chrono;

/**
 g++ -std=c++17  -O3 -DNDEBUG -msse4.2 -mbmi -mbmi2 -Wall -Wextra -pedantic -funroll-loops -D__extern_always_inline="extern __always_inline"  -ffast-math \
   -I/PATHTO/library/sdsl-lite/include -L/PATHTO/library/FM-Index/libdivsufsort/lib \
    -o fm-index-build.exe fm-index-build.cpp -ldivsufsort

*/


int main(int argc, char ** argv)
{
    if (argc < 2)
    {
        cout << "Usage " << argv[0] << " text_file [max_locations] [post_context] [pre_context]" << endl;
        cout << "    This program constructs a very compact FM-index" << endl;
        cout << "    which supports count, locate, and extract queries." << endl;
        cout << "    text_file      Original text file." << endl;
        cout << "    max_locations  Maximal number of location to report." << endl;
        cout << "    post_context   Maximal length of the reported post-context." << endl;
        cout << "    pre_context    Maximal length of the pre-context." << endl;
        return 1;
    }
    size_t max_locations = 5;
    size_t post_context = 10;
    size_t pre_context = 10;
    if (argc >= 3)
    {
        max_locations = atoi(argv[2]);
    }
    if (argc >= 4)
    {
        post_context = atoi(argv[3]);
    }
    if (argc >= 5)
    {
        pre_context = atoi(argv[4]);
    }
    string index_suffix = ".fm9";
    string index_file = string(argv[1]) + index_suffix;
    csa_wt<wt_huff<rrr_vector<127>>, 512, 1024> fm_index;

    if (!load_from_file(fm_index, index_file))
    {
        ifstream in(argv[1]);
        if (!in)
        {
            cout << "ERROR: File " << argv[1] << " does not exist. Exit." << endl;
            return 1;
        }
        cout << "No index " << index_file << " located. Building index now." << endl;
        auto start = high_resolution_clock::now();
        construct(fm_index, argv[1], 1);     // generate index
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<seconds>(stop - start);
        cout << "Index construction took " << duration.count() << " seconds." << endl;
        store_to_file(fm_index, index_file); // save it
    }
    cout << "Index construction complete, index requires " << size_in_mega_bytes(fm_index) << " MiB." << endl;

}
