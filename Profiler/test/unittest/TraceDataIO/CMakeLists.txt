set(FLAGTREE_PROFILER_TEST_UTIL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../util/")
add_compile_definitions(
  FLAGTREE_PROFILER_TEST_UTIL_PATH="${FLAGTREE_PROFILER_TEST_UTIL_PATH}")

add_triton_ut(
	NAME TraceDataIO
	SRCS ByteSpanTest.cpp DecoderTest.cpp CircularLayoutParserTest.cpp ChromeTraceWriterTest.cpp
	LIBS FlagTreeProfilerTraceDataIO
)

target_include_directories(TraceDataIO
PRIVATE
    "${JSON_INCLUDE_DIR}"
	"${PROFILER_COMMON_DIR}/include"
    "${PROFILER_SRC_DIR}/include"
)
