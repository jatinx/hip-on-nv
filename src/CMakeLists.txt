add_library(nvhip64 SHARED
    hipnv.cpp)

target_link_directories(nvhip64 PUBLIC ${CUDA_PATH}/lib64)
target_link_libraries(nvhip64 PUBLIC cudadevrt cudart_static)

install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*")
install(TARGETS nvhip64 
    LIBRARY DESTINATION lib)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../bin/hipcc
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    DESTINATION bin)