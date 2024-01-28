#include <Python.h>

typedef struct
{
    PyObject_VAR_HEAD float **data;
    int rows;
    int cols;
} TensorObject;

static void deallocate(TensorObject *self)
{
    for (int i = 0; i < self->rows; i++)
    {
        free(self->data[i]);
    }
    free(self->data);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int allocate_data(TensorObject *self)
{
    self->data = (float **)malloc(self->rows * sizeof(float *));
    if (self->data == NULL)
    {
        PyErr_NoMemory();
        return -1;
    }
    return 0;
}

static int Tensor_init(TensorObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"data", NULL};

    PyObject *data_arg = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &data_arg))
    {
        return -1;
    }

    PyObject *data_list = PySequence_Fast(data_arg, "expected a sequence");
    if (data_list == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "data must be a sequence");
        return -1;
    }

    self->rows = PySequence_Fast_GET_SIZE(data_list);
    self->cols = 0;

    if (self->rows > 0)
    {
        PyObject *row = PySequence_Fast_GET_ITEM(data_list, 0);
        if (PyList_Check(row))
        {
            self->cols = PySequence_Fast_GET_SIZE(row);
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "data must be a 2D list");
            Py_DECREF(data_list);
            return -1;
        }
    }

    if (allocate_data(self) == -1)
    {
        Py_DECREF(data_list);
        return -1;
    }

    for (int i = 0; i < self->rows; i++)
    {
        PyObject *row = PySequence_Fast_GET_ITEM(data_list, i);
        for (int j = 0; j < self->cols; j++)
        {
            PyObject *item = PySequence_Fast_GET_ITEM(row, j);
            self->data[i][j] = PyFloat_AsDouble(item);
            if (PyErr_Occurred())
            {
                PyErr_SetString(PyExc_TypeError, "elements in data must be numbers");
                Py_DECREF(data_list);
                return -1;
            }
        }
    }

    Py_DECREF(data_list);
    return 0;
}

static PyObject *Tensor_str(TensorObject *self){
    PyObject *result = PyUnicode_FromFormat("Tensor(%d%d)", self->rows, self->cols);
    return result;
}

static PyObject *Tensor_repr(TensorObject *self) {
    PyObject *result = PyUnicode_FromFormat("Tensor(%dx%d, %p)", self->rows, self->cols, (void *)self);
    return result;
}

static PyMethodDef Tensor_methods[] = {
    {NULL, NULL, 0, NULL} // Sentinel
};

static PyTypeObject TensorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "myModule.Tensor",
    .tp_doc = "Tensor class",
    .tp_basicsize = sizeof(TensorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)Tensor_init,
    .tp_dealloc = (destructor)deallocate,
    .tp_str = (reprfunc)Tensor_str,
    .tp_repr = (reprfunc)Tensor_repr,
    .tp_methods = Tensor_methods,
};

static PyModuleDef myModule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "myModule",
    .m_doc = "Example module",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit_myModule(void) {
    PyObject *m;

    if (PyType_Ready(&TensorType) < 0)
        return NULL;

    m = PyModule_Create(&myModule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&TensorType);
    PyModule_AddObject(m, "Tensor", (PyObject *)&TensorType);

    return m;
}