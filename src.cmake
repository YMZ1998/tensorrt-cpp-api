file(
    GLOB TRTCPP_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/*.cpp"
)

set(TRTCPP_CORE_SOURCES ${TRTCPP_SOURCES})
list(FILTER TRTCPP_CORE_SOURCES EXCLUDE REGEX ".*/(oct_segmentation|opencv_interop)\\.cpp$")

set(TRTCPP_OPENCV_SOURCES ${TRTCPP_SOURCES})
list(FILTER TRTCPP_OPENCV_SOURCES INCLUDE REGEX ".*/(oct_segmentation|opencv_interop)\\.cpp$")
