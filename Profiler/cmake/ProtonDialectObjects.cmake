include_guard(GLOBAL)

# Nested backend builds that link object files directly use this list without
# duplicating the Profiler's internal Proton target layout in FlagTree.
function(flagtree_append_profiler_dialect_objects output_var binary_dir)
  set(_root "${binary_dir}/third_party/FlagPrism/Profiler/Dialect")
  set(_objects
    "${_root}/lib/Analysis/CMakeFiles/ProtonAnalysis.dir/ScopeIdAllocation.cpp.o"
    "${_root}/lib/Dialect/ProtonGPU/IR/CMakeFiles/ProtonGPUIR.dir/Dialect.cpp.o"
    "${_root}/lib/Dialect/ProtonGPU/IR/CMakeFiles/ProtonGPUIR.dir/Ops.cpp.o"
    "${_root}/lib/Dialect/ProtonGPU/IR/CMakeFiles/ProtonGPUIR.dir/Types.cpp.o"
    "${_root}/lib/Dialect/ProtonGPU/Transforms/CMakeFiles/ProtonGPUTransforms.dir/ProtonGPUTransformsPass.cpp.o"
    "${_root}/lib/Dialect/Proton/IR/CMakeFiles/ProtonIR.dir/Dialect.cpp.o"
    "${_root}/lib/Dialect/Proton/IR/CMakeFiles/ProtonIR.dir/Ops.cpp.o"
    "${_root}/lib/ProtonGPUToLLVM/CMakeFiles/ProtonGPUToLLVM.dir/AllocateProtonGlobalScratchBuffer.cpp.o"
    "${_root}/lib/ProtonGPUToLLVM/CMakeFiles/ProtonGPUToLLVM.dir/AllocateProtonSharedMemory.cpp.o"
    "${_root}/lib/ProtonGPUToLLVM/CMakeFiles/ProtonGPUToLLVM.dir/PatternProtonGPUOpToLLVM.cpp.o"
    "${_root}/lib/ProtonGPUToLLVM/CMakeFiles/ProtonGPUToLLVM.dir/Utility.cpp.o"
    "${_root}/lib/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/CMakeFiles/ProtonAMDGPUToLLVM.dir/AddSchedBarriers.cpp.o"
    "${_root}/lib/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/CMakeFiles/ProtonAMDGPUToLLVM.dir/AMDPatternProtonGPUOpToLLVM.cpp.o"
    "${_root}/lib/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/CMakeFiles/ProtonAMDGPUToLLVM.dir/ConvertProtonGPUToLLVM.cpp.o"
    "${_root}/lib/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/CMakeFiles/ProtonAMDGPUToLLVM.dir/TargetInfo.cpp.o"
    "${_root}/lib/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/CMakeFiles/ProtonNVIDIAGPUToLLVM.dir/ConvertProtonGPUToLLVM.cpp.o"
    "${_root}/lib/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/CMakeFiles/ProtonNVIDIAGPUToLLVM.dir/NvidiaPatternProtonGPUOpToLLVM.cpp.o"
    "${_root}/lib/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/CMakeFiles/ProtonNVIDIAGPUToLLVM.dir/TargetInfo.cpp.o"
    "${_root}/lib/ProtonToProtonGPU/CMakeFiles/ProtonToProtonGPU.dir/ProtonToProtonGPUPass.cpp.o"
  )
  set(${output_var} ${${output_var}} ${_objects} PARENT_SCOPE)
endfunction()
