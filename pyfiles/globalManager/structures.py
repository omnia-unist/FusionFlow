from ctypes import Structure, c_int32, c_int64


class BOOL_STRUCT(Structure):
    _fields_ = [
        ('shape', c_int32),
        ('size', c_int64),
        ('count', c_int64)
    ]

class INFO_STRUCT(Structure):
    _fields_ = [
        ('shape', c_int32),
        ('size', c_int64),
        ('count', c_int64)
    ]
