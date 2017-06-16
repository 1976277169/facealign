project(facealign)
cmake_minimum_required(VERSION 2.8)

find_package( OpenCV REQUIRED )

set(BOOST_ROOT $ENV{BOOST_ROOT})
set(CMAKE_LIBRARY_PATH ${BOOST_ROOT}/stage/lib)
find_package (Boost REQUIRED)

if (Boost_FOUND)
  # From the offical documentation:
  # Add include directories to the build. [...] If the SYSTEM option is given,
  # the compiler will be told the directories are meant as system include
  # directories on some platforms (signalling this setting might achieve effects
  # such as the compiler skipping warnings [...])."
  include_directories (SYSTEM ${Boost_INCLUDE_DIR})
  message(status "** Boost Include: ${Boost_INCLUDE_DIR}")
  message(status "** Boost Libraries: ${Boost_LIBRARY_DIRS}")
  message(status "** Boost Libraries: ${Boost_LIBRARIES}")
  # From the offical documentation:
  # "Specify directories in which the linker will look for libraries. [...] Note
  # that this command is rarely necessary. Library locations returned by
  # find_package() and find_library() are absolute paths. Pass these absolute
  # library file paths directly to the target_link_libraries() command. CMake
  # will ensure the linker finds them."
  #link_directories (${Boost_LIBRARY_DIRS})
endif ()




include_directories(../include)


#if you don't want to use openMP, you should comment this line
# FIND_PACKAGE( OpenMP REQUIRED)
# if(OPENMP_FOUND)
    # message("OPENMP FOUND")
    # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_C_FLAGS}")
    # set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# endif()
file(GLOB SRC_FILES "*.c*" "../include/*.h*")
add_executable(facealign ${SRC_FILES})
target_link_libraries(facealign ${OpenCV_LIBS} ${Boost_LIBS}) 
