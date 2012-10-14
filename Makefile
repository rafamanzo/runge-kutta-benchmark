# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/rafael/workspace/runge-kutta-benchmark

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/rafael/workspace/runge-kutta-benchmark

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running interactive CMake command-line interface..."
	/usr/bin/cmake -i .
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/rafael/workspace/runge-kutta-benchmark/CMakeFiles /home/rafael/workspace/runge-kutta-benchmark/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/rafael/workspace/runge-kutta-benchmark/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named RungeKuttaBenchmark

# Build rule for target.
RungeKuttaBenchmark: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 RungeKuttaBenchmark
.PHONY : RungeKuttaBenchmark

# fast build rule for target.
RungeKuttaBenchmark/fast:
	$(MAKE) -f CMakeFiles/RungeKuttaBenchmark.dir/build.make CMakeFiles/RungeKuttaBenchmark.dir/build
.PHONY : RungeKuttaBenchmark/fast

Benchmarker.o: Benchmarker.cpp.o
.PHONY : Benchmarker.o

# target to build an object file
Benchmarker.cpp.o:
	$(MAKE) -f CMakeFiles/RungeKuttaBenchmark.dir/build.make CMakeFiles/RungeKuttaBenchmark.dir/Benchmarker.cpp.o
.PHONY : Benchmarker.cpp.o

Benchmarker.i: Benchmarker.cpp.i
.PHONY : Benchmarker.i

# target to preprocess a source file
Benchmarker.cpp.i:
	$(MAKE) -f CMakeFiles/RungeKuttaBenchmark.dir/build.make CMakeFiles/RungeKuttaBenchmark.dir/Benchmarker.cpp.i
.PHONY : Benchmarker.cpp.i

Benchmarker.s: Benchmarker.cpp.s
.PHONY : Benchmarker.s

# target to generate assembly for a file
Benchmarker.cpp.s:
	$(MAKE) -f CMakeFiles/RungeKuttaBenchmark.dir/build.make CMakeFiles/RungeKuttaBenchmark.dir/Benchmarker.cpp.s
.PHONY : Benchmarker.cpp.s

Main.o: Main.cpp.o
.PHONY : Main.o

# target to build an object file
Main.cpp.o:
	$(MAKE) -f CMakeFiles/RungeKuttaBenchmark.dir/build.make CMakeFiles/RungeKuttaBenchmark.dir/Main.cpp.o
.PHONY : Main.cpp.o

Main.i: Main.cpp.i
.PHONY : Main.i

# target to preprocess a source file
Main.cpp.i:
	$(MAKE) -f CMakeFiles/RungeKuttaBenchmark.dir/build.make CMakeFiles/RungeKuttaBenchmark.dir/Main.cpp.i
.PHONY : Main.cpp.i

Main.s: Main.cpp.s
.PHONY : Main.s

# target to generate assembly for a file
Main.cpp.s:
	$(MAKE) -f CMakeFiles/RungeKuttaBenchmark.dir/build.make CMakeFiles/RungeKuttaBenchmark.dir/Main.cpp.s
.PHONY : Main.cpp.s

core/DataSet.o: core/DataSet.cpp.o
.PHONY : core/DataSet.o

# target to build an object file
core/DataSet.cpp.o:
	$(MAKE) -f CMakeFiles/RungeKuttaBenchmark.dir/build.make CMakeFiles/RungeKuttaBenchmark.dir/core/DataSet.cpp.o
.PHONY : core/DataSet.cpp.o

core/DataSet.i: core/DataSet.cpp.i
.PHONY : core/DataSet.i

# target to preprocess a source file
core/DataSet.cpp.i:
	$(MAKE) -f CMakeFiles/RungeKuttaBenchmark.dir/build.make CMakeFiles/RungeKuttaBenchmark.dir/core/DataSet.cpp.i
.PHONY : core/DataSet.cpp.i

core/DataSet.s: core/DataSet.cpp.s
.PHONY : core/DataSet.s

# target to generate assembly for a file
core/DataSet.cpp.s:
	$(MAKE) -f CMakeFiles/RungeKuttaBenchmark.dir/build.make CMakeFiles/RungeKuttaBenchmark.dir/core/DataSet.cpp.s
.PHONY : core/DataSet.cpp.s

core/Fiber.o: core/Fiber.cpp.o
.PHONY : core/Fiber.o

# target to build an object file
core/Fiber.cpp.o:
	$(MAKE) -f CMakeFiles/RungeKuttaBenchmark.dir/build.make CMakeFiles/RungeKuttaBenchmark.dir/core/Fiber.cpp.o
.PHONY : core/Fiber.cpp.o

core/Fiber.i: core/Fiber.cpp.i
.PHONY : core/Fiber.i

# target to preprocess a source file
core/Fiber.cpp.i:
	$(MAKE) -f CMakeFiles/RungeKuttaBenchmark.dir/build.make CMakeFiles/RungeKuttaBenchmark.dir/core/Fiber.cpp.i
.PHONY : core/Fiber.cpp.i

core/Fiber.s: core/Fiber.cpp.s
.PHONY : core/Fiber.s

# target to generate assembly for a file
core/Fiber.cpp.s:
	$(MAKE) -f CMakeFiles/RungeKuttaBenchmark.dir/build.make CMakeFiles/RungeKuttaBenchmark.dir/core/Fiber.cpp.s
.PHONY : core/Fiber.cpp.s

core/cpp/RKCKernel.o: core/cpp/RKCKernel.cpp.o
.PHONY : core/cpp/RKCKernel.o

# target to build an object file
core/cpp/RKCKernel.cpp.o:
	$(MAKE) -f CMakeFiles/RungeKuttaBenchmark.dir/build.make CMakeFiles/RungeKuttaBenchmark.dir/core/cpp/RKCKernel.cpp.o
.PHONY : core/cpp/RKCKernel.cpp.o

core/cpp/RKCKernel.i: core/cpp/RKCKernel.cpp.i
.PHONY : core/cpp/RKCKernel.i

# target to preprocess a source file
core/cpp/RKCKernel.cpp.i:
	$(MAKE) -f CMakeFiles/RungeKuttaBenchmark.dir/build.make CMakeFiles/RungeKuttaBenchmark.dir/core/cpp/RKCKernel.cpp.i
.PHONY : core/cpp/RKCKernel.cpp.i

core/cpp/RKCKernel.s: core/cpp/RKCKernel.cpp.s
.PHONY : core/cpp/RKCKernel.s

# target to generate assembly for a file
core/cpp/RKCKernel.cpp.s:
	$(MAKE) -f CMakeFiles/RungeKuttaBenchmark.dir/build.make CMakeFiles/RungeKuttaBenchmark.dir/core/cpp/RKCKernel.cpp.s
.PHONY : core/cpp/RKCKernel.cpp.s

fixtures/Fixture.o: fixtures/Fixture.cpp.o
.PHONY : fixtures/Fixture.o

# target to build an object file
fixtures/Fixture.cpp.o:
	$(MAKE) -f CMakeFiles/RungeKuttaBenchmark.dir/build.make CMakeFiles/RungeKuttaBenchmark.dir/fixtures/Fixture.cpp.o
.PHONY : fixtures/Fixture.cpp.o

fixtures/Fixture.i: fixtures/Fixture.cpp.i
.PHONY : fixtures/Fixture.i

# target to preprocess a source file
fixtures/Fixture.cpp.i:
	$(MAKE) -f CMakeFiles/RungeKuttaBenchmark.dir/build.make CMakeFiles/RungeKuttaBenchmark.dir/fixtures/Fixture.cpp.i
.PHONY : fixtures/Fixture.cpp.i

fixtures/Fixture.s: fixtures/Fixture.cpp.s
.PHONY : fixtures/Fixture.s

# target to generate assembly for a file
fixtures/Fixture.cpp.s:
	$(MAKE) -f CMakeFiles/RungeKuttaBenchmark.dir/build.make CMakeFiles/RungeKuttaBenchmark.dir/fixtures/Fixture.cpp.s
.PHONY : fixtures/Fixture.cpp.s

fixtures/cpp/CStraightFixture.o: fixtures/cpp/CStraightFixture.cpp.o
.PHONY : fixtures/cpp/CStraightFixture.o

# target to build an object file
fixtures/cpp/CStraightFixture.cpp.o:
	$(MAKE) -f CMakeFiles/RungeKuttaBenchmark.dir/build.make CMakeFiles/RungeKuttaBenchmark.dir/fixtures/cpp/CStraightFixture.cpp.o
.PHONY : fixtures/cpp/CStraightFixture.cpp.o

fixtures/cpp/CStraightFixture.i: fixtures/cpp/CStraightFixture.cpp.i
.PHONY : fixtures/cpp/CStraightFixture.i

# target to preprocess a source file
fixtures/cpp/CStraightFixture.cpp.i:
	$(MAKE) -f CMakeFiles/RungeKuttaBenchmark.dir/build.make CMakeFiles/RungeKuttaBenchmark.dir/fixtures/cpp/CStraightFixture.cpp.i
.PHONY : fixtures/cpp/CStraightFixture.cpp.i

fixtures/cpp/CStraightFixture.s: fixtures/cpp/CStraightFixture.cpp.s
.PHONY : fixtures/cpp/CStraightFixture.s

# target to generate assembly for a file
fixtures/cpp/CStraightFixture.cpp.s:
	$(MAKE) -f CMakeFiles/RungeKuttaBenchmark.dir/build.make CMakeFiles/RungeKuttaBenchmark.dir/fixtures/cpp/CStraightFixture.cpp.s
.PHONY : fixtures/cpp/CStraightFixture.cpp.s

timers/CTimer.o: timers/CTimer.cpp.o
.PHONY : timers/CTimer.o

# target to build an object file
timers/CTimer.cpp.o:
	$(MAKE) -f CMakeFiles/RungeKuttaBenchmark.dir/build.make CMakeFiles/RungeKuttaBenchmark.dir/timers/CTimer.cpp.o
.PHONY : timers/CTimer.cpp.o

timers/CTimer.i: timers/CTimer.cpp.i
.PHONY : timers/CTimer.i

# target to preprocess a source file
timers/CTimer.cpp.i:
	$(MAKE) -f CMakeFiles/RungeKuttaBenchmark.dir/build.make CMakeFiles/RungeKuttaBenchmark.dir/timers/CTimer.cpp.i
.PHONY : timers/CTimer.cpp.i

timers/CTimer.s: timers/CTimer.cpp.s
.PHONY : timers/CTimer.s

# target to generate assembly for a file
timers/CTimer.cpp.s:
	$(MAKE) -f CMakeFiles/RungeKuttaBenchmark.dir/build.make CMakeFiles/RungeKuttaBenchmark.dir/timers/CTimer.cpp.s
.PHONY : timers/CTimer.cpp.s

timers/Timer.o: timers/Timer.cpp.o
.PHONY : timers/Timer.o

# target to build an object file
timers/Timer.cpp.o:
	$(MAKE) -f CMakeFiles/RungeKuttaBenchmark.dir/build.make CMakeFiles/RungeKuttaBenchmark.dir/timers/Timer.cpp.o
.PHONY : timers/Timer.cpp.o

timers/Timer.i: timers/Timer.cpp.i
.PHONY : timers/Timer.i

# target to preprocess a source file
timers/Timer.cpp.i:
	$(MAKE) -f CMakeFiles/RungeKuttaBenchmark.dir/build.make CMakeFiles/RungeKuttaBenchmark.dir/timers/Timer.cpp.i
.PHONY : timers/Timer.cpp.i

timers/Timer.s: timers/Timer.cpp.s
.PHONY : timers/Timer.s

# target to generate assembly for a file
timers/Timer.cpp.s:
	$(MAKE) -f CMakeFiles/RungeKuttaBenchmark.dir/build.make CMakeFiles/RungeKuttaBenchmark.dir/timers/Timer.cpp.s
.PHONY : timers/Timer.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... RungeKuttaBenchmark"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... Benchmarker.o"
	@echo "... Benchmarker.i"
	@echo "... Benchmarker.s"
	@echo "... Main.o"
	@echo "... Main.i"
	@echo "... Main.s"
	@echo "... core/DataSet.o"
	@echo "... core/DataSet.i"
	@echo "... core/DataSet.s"
	@echo "... core/Fiber.o"
	@echo "... core/Fiber.i"
	@echo "... core/Fiber.s"
	@echo "... core/cpp/RKCKernel.o"
	@echo "... core/cpp/RKCKernel.i"
	@echo "... core/cpp/RKCKernel.s"
	@echo "... fixtures/Fixture.o"
	@echo "... fixtures/Fixture.i"
	@echo "... fixtures/Fixture.s"
	@echo "... fixtures/cpp/CStraightFixture.o"
	@echo "... fixtures/cpp/CStraightFixture.i"
	@echo "... fixtures/cpp/CStraightFixture.s"
	@echo "... timers/CTimer.o"
	@echo "... timers/CTimer.i"
	@echo "... timers/CTimer.s"
	@echo "... timers/Timer.o"
	@echo "... timers/Timer.i"
	@echo "... timers/Timer.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

