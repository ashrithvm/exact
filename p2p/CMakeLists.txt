if (NOT MPI_FOUND)
    find_package(MPI)
endif()

if (MPI_FOUND)
    set (CMAKE_CXX_COMPILE_FLAGS "${CMAKE_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS}")
    set (CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")
    include_directories(${MPI_INCLUDE_PATH})

    add_executable(examm_p2p
        examm_p2p.cxx
        peer_ring.cxx
        p2p_comm.cxx
        replica_store.cxx
        p2p_island_strategy.cxx
        p2p_examm.cxx
    )

    # Ensure MPI include directories are available for P2P target
    target_include_directories(examm_p2p PRIVATE ${MPI_INCLUDE_PATH} ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(examm_p2p examm_strategy exact_time_series exact_common exact_weights examm_nn ${MPI_LIBRARIES} ${MPI_CXX_LIBRARIES} ${MPI_EXTRA} ${MYSQL_LIBRARIES} ${TIFF_LIBRARIES} pthread)

endif (MPI_FOUND)
