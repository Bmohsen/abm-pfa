cmake_minimum_required(VERSION 3.14)
project(abmPfa)

# Use C++17 or newer
set(CMAKE_CXX_STANDARD 17)

# 1. Add FastText
add_subdirectory(fastText)

# 2. Add Asio Includes (Fixes 'asio.hpp' not found)
include_directories(${CMAKE_SOURCE_DIR}/asio/include)

# 3. Add Current Directory (for crow_all.h)
include_directories(${CMAKE_SOURCE_DIR})

# 4. Create Executable
add_executable(ai_service main.cpp)

# 5. Link Libraries
if(WIN32)
    # Windows-specific networking libraries
    target_link_libraries(ai_service fasttext-static ws2_32 mswsock)
else()
    # Linux/Mac
    target_link_libraries(ai_service fasttext-static_pic pthread)
endif()