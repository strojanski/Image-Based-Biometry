######################################################################
# Automatically generated by qmake (2.00a) ven 24. giu 14:14:20 2005
######################################################################

TARGET = trimesh_base
LIBPATH += 
DEPENDPATH += . 
INCLUDEPATH += . ../../..
CONFIG += console stl
TEMPLATE = app
SOURCES += trimesh_base.cpp ../../../wrap/ply/plylib.cpp
# Mac specific Config required to avoid to make application bundles
CONFIG -= app_bundle