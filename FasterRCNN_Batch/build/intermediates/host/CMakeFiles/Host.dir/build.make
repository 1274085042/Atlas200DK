# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

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
CMAKE_COMMAND = /home/lfq/MindStudio-ubuntu/tools/cmake/bin/cmake

# The command to remove a file.
RM = /home/lfq/MindStudio-ubuntu/tools/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lfq/AscendProjects/sample-objectdetection/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lfq/AscendProjects/sample-objectdetection/build/intermediates/host

# Include any dependencies generated for this target.
include CMakeFiles/Host.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Host.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Host.dir/flags.make

CMakeFiles/Host.dir/general_image/general_image.cpp.o: CMakeFiles/Host.dir/flags.make
CMakeFiles/Host.dir/general_image/general_image.cpp.o: /home/lfq/AscendProjects/sample-objectdetection/src/general_image/general_image.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lfq/AscendProjects/sample-objectdetection/build/intermediates/host/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Host.dir/general_image/general_image.cpp.o"
	/usr/bin/aarch64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Host.dir/general_image/general_image.cpp.o -c /home/lfq/AscendProjects/sample-objectdetection/src/general_image/general_image.cpp

CMakeFiles/Host.dir/general_image/general_image.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Host.dir/general_image/general_image.cpp.i"
	/usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lfq/AscendProjects/sample-objectdetection/src/general_image/general_image.cpp > CMakeFiles/Host.dir/general_image/general_image.cpp.i

CMakeFiles/Host.dir/general_image/general_image.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Host.dir/general_image/general_image.cpp.s"
	/usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lfq/AscendProjects/sample-objectdetection/src/general_image/general_image.cpp -o CMakeFiles/Host.dir/general_image/general_image.cpp.s

CMakeFiles/Host.dir/general_post/general_post.cpp.o: CMakeFiles/Host.dir/flags.make
CMakeFiles/Host.dir/general_post/general_post.cpp.o: /home/lfq/AscendProjects/sample-objectdetection/src/general_post/general_post.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lfq/AscendProjects/sample-objectdetection/build/intermediates/host/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/Host.dir/general_post/general_post.cpp.o"
	/usr/bin/aarch64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Host.dir/general_post/general_post.cpp.o -c /home/lfq/AscendProjects/sample-objectdetection/src/general_post/general_post.cpp

CMakeFiles/Host.dir/general_post/general_post.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Host.dir/general_post/general_post.cpp.i"
	/usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lfq/AscendProjects/sample-objectdetection/src/general_post/general_post.cpp > CMakeFiles/Host.dir/general_post/general_post.cpp.i

CMakeFiles/Host.dir/general_post/general_post.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Host.dir/general_post/general_post.cpp.s"
	/usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lfq/AscendProjects/sample-objectdetection/src/general_post/general_post.cpp -o CMakeFiles/Host.dir/general_post/general_post.cpp.s

# Object files for target Host
Host_OBJECTS = \
"CMakeFiles/Host.dir/general_image/general_image.cpp.o" \
"CMakeFiles/Host.dir/general_post/general_post.cpp.o"

# External object files for target Host
Host_EXTERNAL_OBJECTS =

/home/lfq/AscendProjects/sample-objectdetection/build/outputs/libHost.so: CMakeFiles/Host.dir/general_image/general_image.cpp.o
/home/lfq/AscendProjects/sample-objectdetection/build/outputs/libHost.so: CMakeFiles/Host.dir/general_post/general_post.cpp.o
/home/lfq/AscendProjects/sample-objectdetection/build/outputs/libHost.so: CMakeFiles/Host.dir/build.make
/home/lfq/AscendProjects/sample-objectdetection/build/outputs/libHost.so: CMakeFiles/Host.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lfq/AscendProjects/sample-objectdetection/build/intermediates/host/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared library /home/lfq/AscendProjects/sample-objectdetection/build/outputs/libHost.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Host.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Host.dir/build: /home/lfq/AscendProjects/sample-objectdetection/build/outputs/libHost.so

.PHONY : CMakeFiles/Host.dir/build

CMakeFiles/Host.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Host.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Host.dir/clean

CMakeFiles/Host.dir/depend:
	cd /home/lfq/AscendProjects/sample-objectdetection/build/intermediates/host && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lfq/AscendProjects/sample-objectdetection/src /home/lfq/AscendProjects/sample-objectdetection/src /home/lfq/AscendProjects/sample-objectdetection/build/intermediates/host /home/lfq/AscendProjects/sample-objectdetection/build/intermediates/host /home/lfq/AscendProjects/sample-objectdetection/build/intermediates/host/CMakeFiles/Host.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Host.dir/depend

