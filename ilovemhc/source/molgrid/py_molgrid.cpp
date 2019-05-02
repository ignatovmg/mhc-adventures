#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>

#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#define E 2.71828

#define NELE 5

#define FUN1(d, r) (exp( -2.0*(d)*(d) /((r)*(r)) ))

#define FUN2(d, r) (1./(E*E) * ( 4.*((d)*(d)/((r)*(r)) ) - 12.*(d)/(r) + 9.))

#define MYFREE(x) {  \
    free(x);         \
    x = NULL;        \
}

#define MAX(x, y) (x) >= (y) ? (x) : (y)
#define MIN(x, y) (x) <= (y) ? (x) : (y)

#define STRINGIZE(X) #X
#define TOSTRING(X) STRINGIZE(X)
#define MSG_ERR(...) {                                                         \
    sprintf(err, __FILE__ ", line " TOSTRING(__LINE__) ": " __VA_ARGS__);      \
    PyErr_SetString(MolgridError, err);                                        \
}

static PyObject *MolgridError;

static int get_element_index(char* channels, char val) {
    for (int i = 0; i < NELE; i++) {
        if (val == channels[i]) {
            return i;
        }
    }
    return -1;
}

//static std::vector<float> 
static float*
compute_grid(const char* pdb_file, 
            const char* type2properties_file, 
            const char* name2type_file, 
            float bin, int nch, long int* dims) {
    
    char err[500];
    std::map<std::string, std::vector<float>> t2p; 
    std::map<std::string, std::string> n2t;
   
    std::string linebuf;
    std::ifstream infile(type2properties_file);
    
    if (!infile) {
        MSG_ERR("File %s couldn't be opened", type2properties_file);
        return NULL;
    }

    try {
        std::getline(infile, linebuf);

        // read atomic property table
        int id;
        //std::array<float, NCH> ar;
        std::string atom_type;
        while(infile >> id >> atom_type) {
            t2p[atom_type].resize(nch);
            for (int i = 0; i < nch; i++) {
                infile >> t2p[atom_type][i];
            }
        }
        infile.close();
    } catch (int e) {
        MSG_ERR("Exception occured, while reading %s", type2properties_file);
        return NULL;
    }

    // read charmm22 atomname -> atom type table
    infile.open(name2type_file);
    if (!infile) {
        MSG_ERR("File %s couldn't be opened", name2type_file);
        return NULL;
    }
    
    try {
        while(std::getline(infile, linebuf)) {
            n2t[linebuf.substr(0, 8)] = linebuf.substr(9, 4);
        }
        infile.close();
    } catch (int e) {
        MSG_ERR("Exception occured, while reading %s", name2type_file);
        return NULL;
    }

    char elements[NELE] = {'H', 'C', 'N', 'O', 'S'};
    float vdw_radii[NELE] = {1.2, 1.7, 1.55, 1.52, 1.8};
    float rad_mult = 1.5;

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

    // tabulate pseudo-gauss function
    // taken from 10.1021/acs.jcim.6b00740
    std::vector<std::vector<float>> gauss_tab(NELE);
    
    float  stride_tab = 0.1;
    for (int c = 0; c < NELE; c++) {
        int nvals = (int)ceil(vdw_radii[c] * rad_mult / stride_tab) + 1;
        gauss_tab[c].resize(nvals);
        
        float d, r = vdw_radii[c];
        
        for (int i = 0; i < nvals; i++) {
            d = i * stride_tab;
            gauss_tab[c][i] = d < r ? FUN1(d, r) : FUN2(d, r);
        }
    }
    
    dims[0] = (long int)nch*2;
    dims[1] = (long int)ceil((upper[0]-lower[0]) / bin) + 1;
    dims[2] = (long int)ceil((upper[1]-lower[1]) / bin) + 1;
    dims[3] = (long int)ceil((upper[2]-lower[2]) / bin) + 1;

    float xyz[3]; // atomic coordinates
    long total_size = dims[0]*dims[1]*dims[2]*dims[3];
    //std::vector<float> array(total_size, 0); // cnn input
    float* array = (float*)calloc(total_size, sizeof(float));
    
    std::string line;
    infile.open(pdb_file);
    
    if (!infile) {
        MSG_ERR("File %s couldn't be opened", pdb_file);
        return NULL;
    }

    // read only ^ATOM lines
    while (std::getline(infile, line)) {
        if (line.substr(0, 4).compare("ATOM") == 0) {

            char chain  = line[21];
            char ele    = line[13];
            if (line[12] != ' ') {
                ele = line[12];
            }
            int ele_id  = get_element_index(elements, ele);

            if (ele_id == -1) {
                MSG_ERR("Couldn't find element '%c' in table", ele);
                return NULL;
            }

            std::string atomname = line.substr(12, 8);
            std::string typen;
            std::vector<float> ppts;
            //std::cout << atomname << " ";
            try {
                typen = n2t.at(atomname);
            } catch (std::out_of_range e) {
                MSG_ERR("Atom name '%s' not found in the name table", atomname.c_str());
                return NULL;
            }
            //std::cout << typen;
            try {
                ppts = t2p.at(typen);
            } catch (std::out_of_range e) {
                MSG_ERR("Atom type '%s' not found in the property table", typen.c_str());
                return NULL;
            }

            xyz[0] = std::stof(line.substr(30, 8));
            xyz[1] = std::stof(line.substr(38, 8));
            xyz[2] = std::stof(line.substr(46, 8));
            
            auto gauss = gauss_tab[ele_id];
            float rad = vdw_radii[ele_id];
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

            for (int c = 0; c < nch; c++) {
                float weight = ppts[c];
                int channel;
                if (chain == 'A') {
                    channel = c;
                } else if (chain == 'B') {
                    channel = c + nch;
                } else {
                    MSG_ERR("Chain must be either A (receptor) or B (ligand). Chain '%c' provided instead", chain);
                    return NULL;
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
    
    return array;
}

static PyObject* make_grid(PyObject *self, PyObject *args)
{
    //const char* command;
    const char* pdb_file; // .pdb
    const char* type2properties_file; // .pdb
    const char* name2type_file; // .pdb
    float bin; // bin size in angstroms
    int nch; // number of channels
    
    char err[500];

    if (!PyArg_ParseTuple(args, "sssfi", &pdb_file, &type2properties_file, &name2type_file, &bin, &nch)) {
        return NULL;
    }
    
    long int dims[4];
    float* array = compute_grid(pdb_file, type2properties_file, name2type_file, bin, nch, dims);
    if (array == NULL) {
        return NULL;
    }
    
    PyObject* output = PyArray_SimpleNewFromData(4, dims, NPY_FLOAT, array);
    if (output == NULL) {
        MYFREE(array);
        MSG_ERR("Couldn't convert array to PyObject*");
        return NULL;
    }
    
    PyArray_ENABLEFLAGS((PyArrayObject*)output, NPY_ARRAY_OWNDATA);
    return output; //Py_BuildValue("(N)", output);
}

static PyMethodDef MolgridMethods[] = {
    {"make_grid",  make_grid, METH_VARARGS, "Execute a shell command."},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initmolgrid(void) {
    PyObject* m = Py_InitModule("molgrid", MolgridMethods);
    if (m == NULL) {
        return;
    }
    import_array();
    
    MolgridError = PyErr_NewException("molgrid.MolgridError", NULL, NULL);
    Py_INCREF(MolgridError);
    PyModule_AddObject(m, "MolgridError", MolgridError);
}
