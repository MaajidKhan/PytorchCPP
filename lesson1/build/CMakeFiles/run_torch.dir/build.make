# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/maajid/project_files/pytorch_june6/lesson1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/maajid/project_files/pytorch_june6/lesson1/build

# Include any dependencies generated for this target.
include CMakeFiles/run_torch.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/run_torch.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/run_torch.dir/flags.make

CMakeFiles/run_torch.dir/main.cpp.o: CMakeFiles/run_torch.dir/flags.make
CMakeFiles/run_torch.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/maajid/project_files/pytorch_june6/lesson1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/run_torch.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/run_torch.dir/main.cpp.o -c /home/maajid/project_files/pytorch_june6/lesson1/main.cpp

CMakeFiles/run_torch.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/run_torch.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/maajid/project_files/pytorch_june6/lesson1/main.cpp > CMakeFiles/run_torch.dir/main.cpp.i

CMakeFiles/run_torch.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/run_torch.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/maajid/project_files/pytorch_june6/lesson1/main.cpp -o CMakeFiles/run_torch.dir/main.cpp.s

# Object files for target run_torch
run_torch_OBJECTS = \
"CMakeFiles/run_torch.dir/main.cpp.o"

# External object files for target run_torch
run_torch_EXTERNAL_OBJECTS =

run_torch: CMakeFiles/run_torch.dir/main.cpp.o
run_torch: CMakeFiles/run_torch.dir/build.make
run_torch: /home/maajid/project_files/pytorch_june6/libtorch/lib/libtorch.so
run_torch: /home/maajid/project_files/pytorch_june6/libtorch/lib/libc10.so
run_torch: /home/maajid/project_files/pytorch_june6/libtorch/lib/libc10.so
run_torch: CMakeFiles/run_torch.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/maajid/project_files/pytorch_june6/lesson1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable run_torch"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/run_torch.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/run_torch.dir/build: run_torch

.PHONY : CMakeFiles/run_torch.dir/build

CMakeFiles/run_torch.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/run_torch.dir/cmake_clean.cmake
.PHONY : CMakeFiles/run_torch.dir/clean

CMakeFiles/run_torch.dir/depend:
	cd /home/maajid/project_files/pytorch_june6/lesson1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/maajid/project_files/pytorch_june6/lesson1 /home/maajid/project_files/pytorch_june6/lesson1 /home/maajid/project_files/pytorch_june6/lesson1/build /home/maajid/project_files/pytorch_june6/lesson1/build /home/maajid/project_files/pytorch_june6/lesson1/build/CMakeFiles/run_torch.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/run_torch.dir/depend

