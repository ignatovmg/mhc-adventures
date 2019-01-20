#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>

#include <map>
#include <string>
#include <iostream>
#include <fstream>

#define E 2.71828

#define NCH 7

#define NELE 5

#define FUN1(d, r) (exp( -2.0*(d)*(d) /((r)*(r)) ))

#define FUN2(d, r) (1./(E*E) * ( 4.*((d)*(d)/((r)*(r)) ) - 12.*(d)/(r) + 9.))

#define MYFREE(x) {\
    free(x);       \
    x = NULL;      \
}

#define MAX(x, y) (x) >= (y) ? (x) : (y)

#define MIN(x, y) (x) <= (y) ? (x) : (y)

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

#define STRINGIZE(X) #X
#define TOSTRING(X) STRINGIZE(X)
#define MSG_ERR(...) {                                       \
    fprintf(stderr, KRED "[ERROR] (" __FILE__                \
    ", " TOSTRING(__LINE__) ") : " __VA_ARGS__);             \
    fprintf(stderr, KNRM);                                   \
    exit(EXIT_FAILURE);                                      \
}

#define MSG_WNG(...) {                                       \
    fprintf(stderr, KYEL "[WARNING] (" __FILE__              \
    ", " TOSTRING(__LINE__) ") : " __VA_ARGS__);             \
    fprintf(stderr, KNRM);                                   \
}

#define MSG_INFO(...) {                                      \
    if (verbosity > 0) {                                     \
        fprintf(stdout, KBLU "[INFO] : " __VA_ARGS__);       \
        fprintf(stdout, KNRM);                               \
    }                                                        \
}

int verbosity = 1;

int get_element_index(char* channels, char val) {
    for (int i = 0; i < NELE; i++) {
        if (val == channels[i]) {
            return i;
        }
    }
    return -1;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        MSG_ERR("Not enough arguments\nUsage: %s input.pdb properties.csv names.csv 0.5 output.bin\n", argv[0]);
    }
    
    int argvi = 1;
    char* pdb_file = argv[argvi++]; // .pdb
    char* type2properties_file = argv[argvi++]; // .pdb
    char* name2type_file = argv[argvi++]; // .pdb
    float bin = atof(argv[argvi++]); // bin size in angstroms
    char* out_file = argv[argvi++]; // .bin

    std::map<std::string, std::array<float, 7>> t2p; 
    std::map<std::string, std::string> n2t;
   
    std::string linebuf;
    std::ifstream infile(type2properties_file);
    std::getline(infile, linebuf);

    // read atomic property table
    int id;
    std::array<float, 7> ar;
    std::string atom_type;
    while(infile >> id >> atom_type >> ar[0] >> ar[1] >> ar[2] >> ar[3] >> ar[4] >> ar[5] >> ar[6]) {
        t2p[atom_type] = ar;
    }
    infile.close();

    // read charmm22 atomname -> atom type table
    infile.open(name2type_file);
    while(std::getline(infile, linebuf)) {
        n2t[linebuf.substr(0, 8)] = linebuf.substr(9, 4);
    }
    infile.close();

    char elements[NELE] = {'H', 'C', 'N', 'O', 'S'};
    float vdw_radii[NELE] = {1.2, 1.7, 1.55, 1.52, 1.8};
    float rad_mult = 1.5;
    //float bin = atof(bin_char); //1.0;

    // precomputed maximum borders of peptide 
    // over all available MHC-peptide structures
    float upper[3] = {33.4, 15.171, 18.314};
    float lower[3] = {7.473, 4.334, 7.701};
    float padding  = 5.0; //
    
    upper[0] = ceil(upper[0] + padding);
    upper[1] = ceil(upper[1] + padding);
    upper[2] = ceil(upper[2] + padding);
    lower[0] = floor(lower[0] - padding);
    lower[1] = floor(lower[1] - padding);
    lower[2] = floor(lower[2] - padding);
    
    MSG_INFO("Input PDB %s\n", pdb_file);
    MSG_INFO("Lower bound: (%.3f, %.3f, %.3f)\n", lower[0], lower[1], lower[2]);
    MSG_INFO("Upper bound: (%.3f, %.3f, %.3f)\n", upper[0], upper[1], upper[2]);

    // tabulate pseudo-gauss function
    // taken from here 10.1021/acs.jcim.6b00740
    float* gauss_tab[NELE];
    float  stride_tab = 0.1;
    for (int c = 0; c < NELE; c++) {
        int nvals = (int)ceil(vdw_radii[c] * rad_mult / stride_tab) + 1;
        gauss_tab[c] = (float*)calloc(nvals, sizeof(float));
        float d, r = vdw_radii[c];
        
        for (int i = 0; i < nvals; i++) {
            d = i * stride_tab;
            gauss_tab[c][i] = d < r ? FUN1(d, r) : FUN2(d, r);
        }
    }
    
    long dims[4] = {(int)NCH*2, // number of channels
                    (int)ceil((upper[0]-lower[0]) / bin) + 1, // dimensions of spacial
                    (int)ceil((upper[1]-lower[1]) / bin) + 1, // part of cnn input
                    (int)ceil((upper[2]-lower[2]) / bin) + 1};
    
    MSG_INFO("Output dimensions: (%i, %i, %i, %i)\n", dims[0], dims[1], dims[2], dims[3]);

    float xyz[3]; // atomic coordinates
    long total_size = dims[0]*dims[1]*dims[2]*dims[3];
    float* array = (float*)calloc(total_size, sizeof(float)); // cnn input
    
    MSG_INFO("Output size: %i x %i = %i (bytes) = %.3f (Mb)\n", 
             total_size, 
             sizeof(float), 
             total_size * sizeof(float), 
             total_size * sizeof(float) / 1024. / 1024.);
    
    std::string line;
    infile.open(pdb_file);
    if (!infile.is_open()) {
        MSG_ERR("Couldn't open the file %s\n", pdb_file);
    }

    // read only ^ATOM lines
    while (std::getline(infile, line)) {
        if (line.substr(0, 4).compare("ATOM") == 0) {
            
            char chain  = line[21];
            char ele    = line[77];
            int ele_id  = get_element_index(elements, ele);
            if (ele_id == -1) {
                MSG_WNG("Couldn't find the query element: '%c'\n", ele);
                continue;
            }
            
            std::string atomname = line.substr(12, 8);
            std::string typen;
            std::array<float, 7> ppts;
            //std::cout << atomname << " ";
            try {
                typen = n2t.at(atomname);
            } catch (std::out_of_range e) {
                std::cout << "Atom name '" << atomname << "' not found in the table" << std::endl;
                continue;
            }
            //std::cout << typen;
            try {
                ppts = t2p.at(typen);
            } catch (std::out_of_range e) {
                std::cout << "Atom type '" << typen << "' not found in the property table" << std::endl;
                continue;
            }
            //for (auto rr : ppts) {
            //    std::cout << rr << " ";
            //}
            //std::cout << std::endl;

            xyz[0] = std::stof(line.substr(30, 8));
            xyz[1] = std::stof(line.substr(38, 8));
            xyz[2] = std::stof(line.substr(46, 8));
            //std::cout << xyz[0] << " " << xyz[1] << " " << xyz[2] << std::endl;

            float* gauss = gauss_tab[ele_id];
            float    rad = vdw_radii[ele_id];
            int lcell[3], ucell[3];
            float pos[3];
            float max_dist = rad * rad_mult;

            lcell[0] = MAX((int)ceil((xyz[0] - lower[0] - max_dist) / bin), 0);
            lcell[1] = MAX((int)ceil((xyz[1] - lower[1] - max_dist) / bin), 0);
            lcell[2] = MAX((int)ceil((xyz[2] - lower[2] - max_dist) / bin), 0);
            ucell[0] = MIN((int)floor((xyz[0] - lower[0] + max_dist) / bin), dims[1]-1);
            ucell[1] = MIN((int)floor((xyz[1] - lower[1] + max_dist) / bin), dims[2]-1);
            ucell[2] = MIN((int)floor((xyz[2] - lower[2] + max_dist) / bin), dims[3]-1);

            pos[0] = (xyz[0] - lower[0]) / bin;
            pos[1] = (xyz[1] - lower[1]) / bin;
            pos[2] = (xyz[2] - lower[2]) / bin;

            for (int c = 0; c < NCH; c++) {
                float weight = ppts[c];
                int channel;

                if (chain == 'A') {
                    channel = c;
                } else if (chain == 'B') {
                    channel = c + NCH;
                } else {
                    MSG_ERR("Wrong chain ID (must be A or B): '%c'\n", chain);
                }

                long ptr = channel*dims[1]*dims[2]*dims[3];
                float gauss_val, dist;
                for (int i = lcell[0]; i <= ucell[0]; i++) {
                    for (int j = lcell[1]; j <= ucell[1]; j++) {
                        for (int k = lcell[2]; k <= ucell[2]; k++) {
                            dist = sqrt((pos[0]-i)*(pos[0]-i) + \
                                        (pos[1]-j)*(pos[1]-j) + \
                                        (pos[2]-k)*(pos[2]-k)) * bin;
                            //dist = 0.0; weight=1.0;
                            if (dist > max_dist) {
                                continue;
                            }
                            
                            gauss_val = gauss[(int)round(dist/stride_tab)];
                            array[ptr + (i*dims[2]+j)*dims[3]+k] = weight*gauss_val;
                        }
                    }
                }
            }
        }
    }    
    
    FILE* fp = fopen(out_file, "w");
    fwrite(&total_size, sizeof(long), 1, fp);
    fwrite(dims, sizeof(long), 4, fp);
    fwrite(array, sizeof(float), total_size, fp);
    fclose(fp);
    
    MSG_INFO("Output written to %s\n", out_file);
    MSG_INFO("Output format: total_size (1 x int%i), dimensions (4 x int%i), values (n x float%i)\n",
             sizeof(long), sizeof(long), sizeof(float));
    
    for (int i = 0; i < NELE; i++) {
        MYFREE(gauss_tab[i]);
    }
    MYFREE(array);
    return 0;
}
