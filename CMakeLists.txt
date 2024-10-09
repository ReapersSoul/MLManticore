project(MLManticore)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set language to C++
enable_language(CXX)

#glob all source files ignore the vcpkg_installed directory
file(GLOB_RECURSE HEADERS "*.hpp")
file(GLOB_RECURSE SOURCES "*.cpp")
#filter out the vcpkg_installed directory
list(FILTER HEADERS EXCLUDE REGEX "vcpkg_installed")
list(FILTER SOURCES EXCLUDE REGEX "vcpkg_installed")
#filter out the test directory
list(FILTER HEADERS EXCLUDE REGEX "test")
list(FILTER SOURCES EXCLUDE REGEX "test")
#filter out GraphicsGorilla directory
list(FILTER HEADERS EXCLUDE REGEX "GraphicsGorilla")
list(FILTER SOURCES EXCLUDE REGEX "GraphicsGorilla")

add_library(${PROJECT_NAME} STATIC ${HEADERS} ${SOURCES})

#set include directory
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/Include/${PROJECT_NAME})
file(MAKE_DIRECTORY ${INCLUDE_DIR})

#copy headers to include directory making sure to preserve directory structure
foreach(header ${HEADERS})
  file(RELATIVE_PATH relPath ${CMAKE_CURRENT_SOURCE_DIR} ${header})
  get_filename_component(dir ${relPath} DIRECTORY)
  file(MAKE_DIRECTORY ${INCLUDE_DIR}/${dir})
  configure_file(${header} ${INCLUDE_DIR}/${relPath} COPYONLY)
endforeach()

add_library(${PROJECT_NAME} STATIC ${HEADERS} ${SOURCES})

#add_library(opencl SHARED IMPORTED)
#set_target_properties(opencl PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/jniLibs/libOpenCL.so)

# find_package(OpenCL CONFIG REQUIRED)
# target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)
# target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::Headers)
# target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::HeadersCpp)

find_package(OpenCL REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)

#link the OpenCL library according to the above snippet
#target_link_libraries(${PROJECT_NAME} opencl)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR})

# test project
add_executable(${PROJECT_NAME}_test test/test.cpp)
target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}_test PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME}_test PUBLIC ${CMAKE_SOURCE_DIR}/${PROJECT_NAME})

#glm
find_package(glm CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME}_test PRIVATE glm::glm)
#glew
find_package(GLEW REQUIRED)
target_link_libraries(${PROJECT_NAME}_test PRIVATE GLEW::GLEW)
#glfw
find_package(glfw3 CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME}_test PRIVATE glfw)

add_subdirectory(test/GraphicsGorilla)
target_link_libraries(${PROJECT_NAME}_test PRIVATE GraphicsGorilla)