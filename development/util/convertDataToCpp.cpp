#include <vector>
#include <iostream>
#include <iterator>
#include <fstream>
#include <algorithm>
#include <string>

#include <Python.h>

static PyObject* parseAndStoreVector(PyObject* self, PyObject* args) {
    const char* fileName;
    PyObject *instancesListObj, *featuresListObj, *dataListObj;
    size_t maxNumberOfInstances, maxNumberOfFeatures;
    if (!PyArg_ParseTuple(args, "O!O!O!kks", 
                            &PyList_Type, &instancesListObj, 
                            &PyList_Type, &featuresListObj,
                            &PyList_Type, &dataListObj,
                            &maxNumberOfInstances, &maxNumberOfFeatures,
                            &fileName))
        return NULL;
    std::vector<size_t> instances;
    std::vector<size_t> features;
    std::vector<float> data;
    std::string fileNameStr(fileName);
    std::string instancesString("_instances");
    std::string featuresString("_features");
    std::string dataString("_data");
    std::string additionalInformationString("_addInfo");

    size_t sizeOfFeatureVector = PyList_Size(instancesListObj);
    size_t featureValue;
    size_t instanceValue;
    float dataValue;
    PyObject* instanceSize_tObj;
    PyObject* featureSize_tObj;
    PyObject* dataSize_tObj;
    for (size_t i = 0; i < sizeOfFeatureVector; ++i) {
        instanceSize_tObj = PyList_GetItem(instancesListObj, i);
        featureSize_tObj = PyList_GetItem(featuresListObj, i);
        dataSize_tObj = PyList_GetItem(dataListObj, i); 
        
        PyArg_Parse(instanceSize_tObj, "k", &instanceValue);
        PyArg_Parse(featureSize_tObj, "k", &featureValue);
        PyArg_Parse(dataSize_tObj, "f", &dataValue);

        instances.push_back(instanceValue);
        features.push_back(featureValue);
        data.push_back(dataValue);
    }

    std::fstream file_instances(fileName+instancesString, std::ios::out);
    std::fstream file_features(fileName+featuresString, std::ios::out);
    std::fstream file_data(fileName+dataString, std::ios::out);

    if(!file_instances)
        return Py_BuildValue("i", 1);
    if(!file_features)
        return Py_BuildValue("i", 1);
    if(!file_data)
        return Py_BuildValue("i", 1);
    std::copy(instances.begin(),instances.end(), std::ostream_iterator<size_t>(file_instances," "));
    std::copy(features.begin(),features.end(), std::ostream_iterator<size_t>(file_features," "));
    std::copy(data.begin(),data.end(), std::ostream_iterator<float>(file_data," "));

    std::ofstream additionalInformation;
    additionalInformation.open (fileName+additionalInformationString);
    additionalInformation << maxNumberOfInstances << " " << maxNumberOfFeatures;
    additionalInformation.close();
    return Py_BuildValue("i", 0);

}

// definition of avaible functions for python and which function parsing fucntion in c++ should be called.
static PyMethodDef parse[] = {
    {"parseAndStoreVector", parseAndStoreVector, METH_VARARGS, "Parse the given dataset and store it in a file."},
    {NULL, NULL, 0, NULL}
};

// definition of the module for python
PyMODINIT_FUNC
init_convertDataToCpp(void)
{
    (void) Py_InitModule("_convertDataToCpp", parse);
}
