project(iCell)
cmake_minimum_required(VERSION 2.8)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
if (ITKVtkGlue_LOADED)
  find_package(VTK REQUIRED)
  include(${VTK_USE_FILE})
else()
  find_package(ItkVtkGlue REQUIRED)
  include(${ItkVtkGlue_USE_FILE})
  set(Glue ItkVtkGlue)
endif()

INCLUDE_DIRECTORIES(Library)

SET(ICELL_COMMON_SRC
    Library/forest.h
    Library/classification.h
    Library/classifier.h
    Library/data.h
    Library/ImageCollectionToImageFilter.h
    Library/imageio.h
    Library/linearalgebra.h
    Library/node.h
    Library/random.h
    Library/statistics.h
    Library/trainer.h
    Library/trainingcontext.h
    Library/tree.h
    Library/type.h
    Library/utility.h
    )
SET(ICELL_TRAIN_SRC
    Library/RFsample.h
    Library/RFsample.txx
    )
SET(ICELL_APPLY_SRC
    Library/RFapply.h
    Library/RFapply.txx
    )

SET(ICELLTRAIN_SRC
   main.cpp
   ${ICELL_COMMON_SRC} ${ICELL_TRAIN_SRC})

SET(ICELLAPPLY_SRC
   apply_main.cpp
   ${ICELL_COMMON_SRC} ${ICELL_APPLY_SRC})

add_executable(icell_train ${ICELLTRAIN_SRC})
target_link_libraries(icell_train ${GLUE} ${ITK_LIBRARIES} ${VTK_LIBRARIES})

add_executable(icell_apply ${ICELLAPPLY_SRC})
target_link_libraries(icell_apply ${GLUE} ${ITK_LIBRARIES} ${VTK_LIBRARIES})
